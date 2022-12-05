from typing import Optional, List, Tuple
import math
from tqdm import tqdm
from utils.tools import logger
from portfolio_manager.algorithms import *
from environment.environment import Environment
from agent.agent import Agent
from utils.constants import *


class Exp_Main:
    def __init__(self, args) -> None:
        self.args = args

        self.agent = self._set_agent()
        self.train_env = self._set_environment(flag="train")
        self.test_env = self._set_environment(flag="test")
        self.train_benchmark = self.get_benchmark(args.benchmark_name, flag="train")
        self.test_benchmark = self.get_benchmark(args.benchmark_name, flag="test")

        self.initial_value = START_VALUE

        self.train_scores_episodes = []
        self.test_scores_episodes = []
        self.train_action_histories = []
        self.test_action_histories = []

        self.names = [
            "train_scores_episodes",
            "test_scores_episodes",
            "train_action_histories",
            "test_action_histories",
        ]
        self.paths = [os.path.join("outputs/results", name) for name in self.names]
        os.makedirs("outputs/results", exist_ok=True)

    @property
    def get_results(self) -> Tuple[List, List]:
        return (
            self.train_scores_episodes,
            self.test_scores_episodes,
            self.train_action_histories,
            self.test_action_histories,
        )

    def save_results(self):
        for path_name in self.paths:
            if os.path.exists(path_name):
                os.remove(path_name)
        for name, path_name in zip(self.names, self.paths):
            np.save(path_name, np.array(getattr(self, name)))

    def _set_agent(self) -> None:
        return Agent(self.args, flag="train")

    def _set_environment(self, flag) -> None:
        if flag == "train":
            return Environment(self.args, flag="train")
        else:
            return Environment(self.args, flag="test")

    def get_benchmark(self, model_name, flag: str = "train"):
        args = self.args
        model_map = {
            "CRP": CRP,
            "UBAH": UBAH,
            "BCRP": BCRP,
            "BestMarkowitz": BestMarkowitz,
            "UP": UP,
            "Anticor": Anticor,
            "OLMAR": OLMAR,
            "RMR": RMR,
        }
        if model_name not in model_map:
            logger.warn(f"No model named {model_name}")
            return
        model = model_map[model_name](args, flag=flag)
        weights = model.run(model.X)
        return model.calculate_returns(weights)

    def log_episode_result(
        self, episode: int, train_scores: List, test_scores: Optional[List]
    ):
        """Logs the training result after each episode"""
        if train_scores and test_scores:
            train_value = self.initial_value * math.exp(sum(train_scores))
            test_value = (
                self.initial_value * math.exp(sum(test_scores)) if test_scores else 0
            )
            logger.info(
                f"Episode: {episode} --- Train Value: {train_value:.2f} --- Test Value: {test_value:.2f}"
            )

    def log_benchmark(self, in_dollar: bool = True) -> None:
        """Logs the benchmark of the train and test datasat. Specific algorithm is specified under args.bechmark_name"""
        total_return_train = self.train_benchmark.prod()
        total_return_test = self.test_benchmark.prod()
        portfolio_value_train = (
            self.initial_value * total_return_train if in_dollar else total_return_train
        )
        portfolio_value_test = (
            self.initial_value * total_return_test if in_dollar else total_return_test
        )
        logger.info(
            f"Benchmark: {self.args.benchmark_name} --- Train Value: {portfolio_value_train:.2f} - Trading Periods: {self.train_env.num_steps} --- Test Value: {portfolio_value_test:.2f} - Trading Periods: {self.test_env.num_steps}"
        )

    def plot_results(self):
        """Plot the resulting portfolio value after each episode (x=episode) redline = ubah
        plot the portfolio weights of last period? plot portfolio weights ob backtest etc..."""
        raise NotImplementedError()

    def train(self, with_test: bool = False, resume: bool = False) -> None:
        if resume:
            self.agent.load_models()

        test_steps = self.test_env.num_steps - self.args.seq_len if with_test else 0
        self.log_benchmark(in_dollar=True)
        # if self.args.noise == "OU":
        #   self.agent.noise.reset()

        for episode in range(self.args.episodes):
            done = False
            train_scores = []
            obs, train_steps = self.train_env.reset()
            total_steps = train_steps + test_steps - self.args.seq_len
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

                test_scores = self.backtest(bar=pbar) if with_test else None

            self.log_episode_result(
                episode=episode, train_scores=train_scores, test_scores=test_scores
            )

            self.train_scores_episodes.append(train_scores)
            self.test_scores_episodes.append(test_scores)
            self.train_action_histories.append(self.train_env.action_history)

            if episode % 5 == 0 and episode != 0:
                self.agent.save_models()
            self.save_results()

    def backtest(self, bar=None, env=None) -> None:
        score_history = []
        done = False
        env = env or self.test_env
        obs, _ = env.reset()
        while not done:
            act = self.agent.choose_action(obs, flag="test")
            new_state, reward, done = env.step(act)
            # score += reward
            score_history.append(reward)
            obs = new_state
            if bar:
                bar.update(1)

        self.test_action_histories.append(env.action_history)
        return score_history
