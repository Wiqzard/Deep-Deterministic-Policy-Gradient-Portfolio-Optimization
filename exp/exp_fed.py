import time
from typing import List, Union
import numpy as np
from tqdm import tqdm
import torch
from fed_former.agent_ts import Agent
from fed_former.data_factory import DataSet
from fed_former.lstm import ActorLSTM
from torch.utils.data import DataLoader
import torch.optim as optim
from exp.exp_basic import Exp_Basic

from utils.constants import *
from utils.tools import logger
import logging

logging.basicConfig()
logger = logging.getLogger()
#
class Exp_Fed(Exp_Basic):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.train_data = DataSet(args, flag="train")
        self.test_data = DataSet(args, flag="test")

        self.device = "cuda" if args.use_gpu else "cpu"
        self.actor = ActorLSTM(args, embed_type=args.embed_type, freq="t").to(
            self.device
        )

    def get_start_action(self, flag="uni"):
        if flag == "uni":
            return torch.tensor(NUM_ASSETS * [1 / NUM_ASSETS])

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

    def get_optimizer(self) -> Union[optim.Adam, optim.SGD]:
        # sourcery skip: assign-if-exp, inline-immediately-returned-variable
        if self.args.optim == "adam":
            return optim.Adam(self.actor.parameters(), lr=self.args.actor_learning_rate)
        else:
            return None

    def train(self, with_test: bool = True, resume: bool = False) -> None:
        args = self.args
        if resume:
            self.actor.load_checkpoint()
        # self.log_benchmark(in_dollar=True)

        dataloader = self.get_dataloader("train")
        optimizer = self.get_optimizer()
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        total_steps = (
            len(self.train_data) + len(self.test_data)
            if with_test
            else len(self.train_data)
        )

        for episode in range(args.episodes):
            self.actor.train()
            action_history = []

            with tqdm(
                total=total_steps,
                leave=self.args.colab,
                position=1 - int(self.args.colab),
            ) as pbar:
                for i, (idxs, scales, states, prev_actions, _) in enumerate(dataloader):
                    optimizer.zero_grad()
                    states, _, state_time_marks, _ = states
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            actions = self.actor(states, state_time_marks, prev_actions)
                    else:
                        actions = self.actor(states, state_time_marks, prev_actions)
                    rewards = self.calculate_rewards_torch(
                        scales, states, prev_actions, actions, self.args
                    )
                    reward = -sum(rewards)
                    # reward =  -self.calculate_cummulative_reward(rewards)
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

                    train_scores.append(reward.detach().cpu().item())
                    action_history.append(actions.detach().cpu().numpy())
                    self.train_data.action_memory.store_action(
                        actions.detach().cpu().numpy(), idxs
                    )
            # for name, param in self.actor.named_parameters():
            #    print(name, param)
            train_scores = [reward.detach().cpu().numpy()]
            self.actor.save_checkpoint()
            test_scores = self.backtest(bar=pbar) if with_test else None

            self.log_episode_result(
                episode=episode, train_scores=train_scores, test_scores=test_scores
            )
            self.train_scores_episodes.append(train_scores)
            self.test_scores_episodes.append(test_scores)
            self.train_action_histories.append(action_history)

            self.save_results()

    def backtest(self, data=None, bar=None) -> List[float]:
        self.actor.load_checkpoint()
        self.actor.eval()
        score_history = []
        action_history = []
        #        test_dataloader =self.get_datalaoder(flag="custom", data=data) if data else self.get_datalaoder(flag="test")

        test_data = data or self.test_data
        prev_action = self.get_start_action(flag="uni")
        print("test")
        for idx in range(len(test_data)):
            _, scale, state, _, _ = test_data[idx]
            state, _, state_time_mark, _ = state
            state = state.unsqueeze(0)
            state_time_mark = state_time_mark.unsqueeze(0)
            prev_action = prev_action.unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(state, state_time_mark, prev_action)
            if self.args.ba:
                print(action)

            reward = self.calculate_rewards_torch(
                scale, state, prev_action, action, self.args
            )
            reward = reward[-1].cpu().numpy()
            prev_action = action
            score_history.append(reward)
            action_history.append(action.cpu().numpy())
            if bar:
                bar.update(1)

        self.test_action_histories.append(action_history)
        return score_history

    #    def reward(self, model_output):
    # reward = self.calculate_cummulative_reward(
    # self.calculate_rewards_torch(model_output)
    # )
    # return reward

    def calculate_rewards_torch(self, scales, states, prev_actions, actions, args):
        """
        put that into data class
        args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
        -> Calculate y_t from X_t = v_t // v_t-1
        -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
        -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
        -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
        """
        seq_x_s = states
        mus, sigmas = scales
        rewards = []
        for batch in range(seq_x_s.shape[0]):
            mu = mus[batch]
            sigma = sigmas[batch]
            X_t = seq_x_s[batch]
            X_t = np.multiply(X_t, sigma) + mu
            X_t = torch.tensor(X_t, dtype=torch.float32).float().to(self.device)
            # print("X_t:", X_t)
            w_t_1 = prev_actions[batch].float().to(self.device)
            # print("w_t_1:", w_t_1)
            y_t = X_t[args.seq_len - 1, :] / X_t[args.seq_len - 2, :]
            # print("y_t:", y_t)
            w_t_prime = (torch.multiply(y_t, w_t_1)) / torch.dot(y_t, w_t_1)
            # print("w_t_prime:", w_t_prime)
            # print("actions:", actions[batch])
            mu_t = 1 - args.commission_rate_selling * sum(
                torch.abs(w_t_prime - actions[batch])
            )
            # print("mu_t:", mu_t)
            r_t = torch.log(mu_t * torch.dot(y_t, w_t_1))
            # print("r_t:", r_t)
            rewards.append(r_t)
        return rewards

    def calculate_cummulative_reward(self, rewards):
        result = torch.tensor([0])
        for reward in enumerate(rewards):
            reward
        cumm_reward = [element / (i + 1) for i, element in enumerate(rewards)]

        return torch.sum(cumm_reward)

    def log_benchmark(self, in_dollar: bool = True) -> None:
        """Logs the benchmark of the train and test datasat. Specific algorithm is specified under args.bechmark_name"""
        portfolio_value_train = self.ubah(flag="train")
        portfolio_value_test = self.ubah(flag="test")
        logger.info(
            f"Benchmark: {self.args.benchmark_name} --- Train Value: {portfolio_value_train:.2f} - Trading Periods: {len(self.train_data)} --- Test Value: {portfolio_value_test:.2f} - Trading Periods: {len(self.test_data)}"
        )
