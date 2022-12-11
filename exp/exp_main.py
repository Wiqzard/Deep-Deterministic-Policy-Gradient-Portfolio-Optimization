from tqdm import tqdm
from utils.tools import logger
from portfolio_manager.algorithms import *
from environment.environment import Environment
from agent.agent import Agent
from utils.constants import *
from exp_basic import Exp_Basic


class Exp_Main(Exp_Basic):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args

        self.agent = self._set_agent()
        self.train_env = self._set_environment(flag="train")
        self.test_env = self._set_environment(flag="test")

    def _set_agent(self) -> None:
        return Agent(self.args, flag="train")

    def _set_environment(self, flag) -> None:
        if flag == "train":
            return Environment(self.args, flag="train")
        else:
            return Environment(self.args, flag="test")

    def train(self, with_test: bool = False, resume: bool = False) -> None:
        if resume:
            self.agent.load_models()

        test_steps = self.test_env.num_steps - self.args.seq_len - 1 if with_test else 0
        self.log_benchmark(in_dollar=True)

        for episode in range(self.args.episodes):
            done = False
            train_scores = []
            obs, train_steps = self.train_env.reset()
            total_steps = train_steps + test_steps - self.args.seq_len - 1
            with tqdm(
                total=total_steps,
                leave=self.args.colab,
                position=1 - int(self.args.colab),
            ) as pbar:
                while not done:
                    act = self.agent.choose_action(obs)

                    new_state, reward, done = self.train_env.step(act)
                    train_scores.append(reward)
                    self.agent.remember(obs, act, reward, new_state, int(done))
                    self.agent.learn()
                    obs = new_state
                    pbar.update(1)

                self.agent.save_models()
                test_scores = self.backtest(bar=pbar) if with_test else None

            self.log_episode_result(
                episode=episode, train_scores=train_scores, test_scores=test_scores
            )

            self.train_scores_episodes.append(train_scores)
            self.test_scores_episodes.append(test_scores)
            self.train_action_histories.append(self.train_env.action_history)

            self.save_results()

    def backtest(self, bar=None, env=None) -> None:
        self.agent.load_models()

        score_history = []
        done = False
        env = env or self.test_env
        obs, _ = env.reset()
        while not done:
            act = self.agent.choose_action(obs, flag="test")
            new_state, reward, done = env.step(act)
            self.agent.fix_const_outputs()
            obs = new_state

            if self.args.ba:
                print(act)

            score_history.append(reward)
            if bar:
                bar.update(1)

        self.test_action_histories.append(env.action_history)
        return score_history
