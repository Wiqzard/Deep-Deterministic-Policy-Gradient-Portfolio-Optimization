import time
from typing import List, Union
import numpy as np
from tqdm import tqdm
import torch
from fed_former.agent_ts import Agent
from fed_former.data_factory import DataSet
from fed_former.lstm import ActorLSTM, ActorNetwork2
from torch.utils.data import DataLoader
import torch.optim as optim
from exp.exp_basic import Exp_Basic

from utils.constants import *
from utils.tools import logger
import logging

LAMBDA = 1e-4
logging.basicConfig()
logger = logging.getLogger()
#
class Exp_Fed(Exp_Basic):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.train_data = DataSet(args, flag="train")
        self.test_data = DataSet(args, flag="test")
        self.__commission_ratio = args.commission_rate_selling
        self.device = "cuda" if args.use_gpu else "cpu"
        self.actor = ActorLSTM(args, embed_type=args.embed_type, freq="t").to(
            self.device
        )

    def get_start_action(self, flag="uni"):
        if flag == "uni":
            return torch.tensor(NUM_ASSETS * [1 / NUM_ASSETS])

    @property
    def actor_params(self):
        total_params = sum(
            p.numel() for p in self.actor.parameters() if p.requires_grad
        )
        return total_params

    def get_dataloader(self, flag: str, data=None) -> DataLoader:
        args = self.args
        if flag == "custom":
            return DataLoader(
                data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=args.droplast,
                num_workers=args.num_workers,
            )
        elif flag == "test":
            return DataLoader(
                self.test_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=args.drop_last,
                num_workers=args.num_workers,
            )
        elif flag == "train":
            return DataLoader(
                self.train_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=args.drop_last,
                num_workers=args.num_workers,
            )

    def get_optimizer(self, ascend: bool = False) -> Union[optim.Adam, optim.SGD]:
        # sourcery skip: assign-if-exp, inline-immediately-returned-variable
        if self.args.optim == "adam":
            return optim.Adam(
                self.actor.parameters(),
                lr=self.args.actor_learning_rate,
                maximize=ascend,
            )
        else:
            return None

    def get_loss(self, flag="with_w"):
        def crit1(actions):
            return torch.mean(
                torch.log(torch.sum(actions * self.__future_price, dim=1))
            ) - LAMBDA * torch.mean(torch.sum(-torch.log(1 + 1e-6 - actions), dim=1))

        def crit2(actions):
            return torch.mean(
                torch.log(torch.sum(actions[:, :] * self.__future_price[:, :], dim=1))
                - torch.sum(
                    torch.abs(actions[:, :] - self.__previous_w[:, :])
                    * self.__commission_ratio,
                    dim=1,
                )
            )

        def crit3(actions):
            return (
                torch.mean(torch.log(torch.sum(actions * self.__future_price, dim=1)))
                - LAMBDA * torch.mean(torch.sum(-torch.log(1 + 1e-6 - actions), dim=1))
                + self.args.diff_factor
                * torch.sum(torch.abs(actions - self.__previous_w), dim=1)
                / NUM_ASSETS
            )

        if flag == "plain":
            return crit1
        elif flag == "with_w":
            return crit2
        elif flag == "with_diff":
            return crit3

    def __set_future_price(self, states, scales) -> None:
        """y_t from paper"""
        X_t = torch.add(
            torch.multiply(
                states,
                scales[1],
            ),
            scales[0],
        )
        if self.args.y == 1:
            self.__future_price = torch.divide(X_t[:, -1, :], X_t[:, -2, :]).to(
                self.device
            )
        elif self.args.y == 2:
            self.__future_price = torch.divide(X_t[:, -2, :], X_t[:, -3:, :]).to(
                self.device
            )

    def train(self, with_test: bool = True, resume: bool = False) -> None:
        args = self.args
        if resume:
            self.actor.load_checkpoint()
        self.log_benchmark(in_dollar=True)

        dataloader = self.get_dataloader("train")
        optimizer = self.get_optimizer(ascend=True)
        criterion = self.get_loss(args.criterion)  # criterion

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        total_steps = (
            len(self.train_data) + len(self.test_data)
            if with_test
            else len(self.train_data)
        )

        for episode in range(args.episodes):
            self.actor.train()
            self.action_history = []
            self.train_scores = []

            with tqdm(
                total=total_steps,
                leave=self.args.colab,
                position=1 - int(self.args.colab),
            ) as pbar:
                for i, (idxs, scales, states, prev_actions, _) in enumerate(dataloader):

                    states, _, state_time_marks, _ = states
                    self.__previous_w = prev_actions.to(self.device)
                    self.__set_future_price(states, scales)
                    optimizer.zero_grad()
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            actions = self.actor(states, state_time_marks, prev_actions)
                    else:
                        actions = self.actor(states, state_time_marks, prev_actions)

                    if self.args._print_train:
                        print(actions)

                    reward = criterion(actions)
                    start = time.time()
                    if self.args.use_amp:
                        scaler.scale(reward).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        reward.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    end = time.time()
                    # print("backward took %.6f seconds" % (end - start))

                    scores = self.calculate_rewards_torch(actions)
                    self.__store(actions, scores)
                    print(sum(self.train_scores))
                    self.train_data.action_memory.store_action(
                        actions.detach().cpu().numpy(), idxs
                    )
                    pbar.update(args.batch_size)

                self.actor.save_checkpoint()
                test_scores = self.backtest(bar=pbar) if with_test else None

            self.log_episode_result(
                episode=episode, train_scores=self.train_scores, test_scores=test_scores
            )
            self.train_scores_episodes.append(self.train_scores)
            self.test_scores_episodes.append(test_scores)
            self.train_action_histories.append(self.action_history)

            self.save_results()

    def __store(self, actions, scores) -> None:
        """responsible for storing scores and actions for each step"""
        for batch in range(actions.shape[0]):
            self.action_history.append(actions[batch, :].detach().cpu().numpy())
            self.train_scores.append(scores[batch])

    def backtest(self, data=None, bar=None) -> List[float]:
        self.actor.load_checkpoint()
        self.actor.eval()
        score_history = []
        action_history = []
        #        test_dataloader =self.get_datalaoder(flag="custom", data=data) if data else self.get_datalaoder(flag="test")

        test_data = data or self.test_data
        prev_action = self.get_start_action(flag="uni")
        for idx in range(len(test_data)):
            _, scale, state, prev_actions, _ = test_data[idx]
            state, _, state_time_mark, _ = state
            state = state.unsqueeze(0)
            state_time_mark = state_time_mark.unsqueeze(0)
            prev_action = prev_action.unsqueeze(0).to(self.device)

            with torch.no_grad():
                action = self.actor(state, state_time_mark, prev_action)
            if self.args.ba:
                print(action)

            self.__set_future_price(state, scale)
            self.__previous_w = prev_action.to(self.device)

            reward = self.calculate_rewards_torch(action.unsqueeze(0))[-1]
            prev_action = action
            score_history.append(reward)
            action_history.append(action.cpu().numpy())
            if bar:
                bar.update(1)
        self.test_action_histories.append(action_history)
        return score_history

    def calculate_rewards_torch(self, actions):
        """
        put that into data class
        args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
        -> Calculate y_t from X_t = v_t // v_t-1
        -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
        -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
        -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
        """
        c = self.__commission_ratio
        y_t = self.__future_price
        w_t_1 = self.__previous_w
        w_t_prime = torch.multiply(y_t, w_t_1) / torch.sum(
            y_t * w_t_1, dim=1, keepdim=True
        )
        w_t = actions
        mu = 1 - c * torch.sum(torch.abs(w_t_prime - w_t), dim=-1)

        def recurse(mu0):
            factor1 = 1 / (1 - c * w_t_1[:, 0])
            mu0 = mu0 if isinstance(mu0, float) else mu0[:, None]
            factor2 = (
                1
                - c * w_t[:, 0]
                - (2 * c - c**2)
                * torch.nn.functional.relu(
                    torch.sum(w_t[:, 1:] - mu0 * w_t_1[:, 1:], dim=-1)
                )
            )
            return factor1 * factor2

        for i in range(20):
            mu = recurse(mu)
        r_t = torch.log(mu * torch.sum(y_t * w_t_1, dim=1))  # .squeeze()
        rewards = r_t.tolist()
        return rewards[-1]

    def log_benchmark(self, in_dollar: bool = True) -> None:
        """Logs the benchmark of the train and test datasat. Specific algorithm is specified under args.bechmark_name"""
        portfolio_value_train = self.ubah(flag="train")
        portfolio_value_test = self.ubah(flag="test")
        logger.info(
            f"Benchmark: {self.args.benchmark_name} --- Train Value: {portfolio_value_train:.2f} - Trading Periods: {len(self.train_data)} --- Test Value: {portfolio_value_test:.2f} - Trading Periods: {len(self.test_data)}"
        )
