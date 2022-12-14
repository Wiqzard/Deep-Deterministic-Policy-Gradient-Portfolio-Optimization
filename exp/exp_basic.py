from typing import Optional, List, Tuple
import math
from utils.tools import logger
from portfolio_manager.algorithms import *
from utils.constants import *
from utils.tools import train_test_split, add_periods_to_datetime, calculate_returns
from data_management.data_manager import PriceHistory


class Exp_Basic:
    def __init__(self, args) -> None:
        self.args = args

        self.initial_value = START_VALUE

        self.train_benchmark = self.get_benchmark(args.benchmark_name, flag="train")
        self.test_benchmark = self.get_benchmark(args.benchmark_name, flag="test")

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

            if self.args.use_numeraire:
                numeraire_ratio_train = self.train_env.get_numeraire_ratio()
                train_value = self.initial_value * (
                    math.exp(sum(train_scores[1:])) * numeraire_ratio_train
                )
                numeraire_ratio_test = self.test_env.get_numeraire_ratio()
                test_value = self.initial_value * (
                    math.exp(sum(test_scores[1:])) * numeraire_ratio_test
                )
            else:
                train_value = self.initial_value * math.exp(sum(train_scores))
                test_value = (
                    self.initial_value * math.exp(sum(test_scores))
                    if test_scores
                    else 0
                )
            logger.info(
                f"Episode: {episode} --- Train Value: {train_value:.2f} --- Test Value: {test_value:.2f}"
            )

    def log_benchmark(self, in_dollar: bool = True) -> None:
        """Logs the benchmark of the train and test datasat. Specific algorithm is specified under args.bechmark_name"""
        # total_return_train = self.train_benchmark.prod()
        # total_return_test = self.test_benchmark.prod()
        # portfolio_value_train = (
        #     self.initial_value * total_return_train if in_dollar else total_return_train
        # )
        # portfolio_value_test = (
        #     self.initial_value * total_return_test if in_dollar else total_return_test
        # )
        portfolio_value_train = self.ubah(flag="train")
        portfolio_value_test = self.ubah(flag="test")
        logger.info(
            f"Benchmark: {self.args.benchmark_name} --- Train Value: {portfolio_value_train:.2f} - Trading Periods: {self.train_env.num_steps} --- Test Value: {portfolio_value_test:.2f} - Trading Periods: {self.test_env.num_steps}"
        )

    def plot_results(self):
        """Plot the resulting portfolio value after each episode (x=episode) redline = ubah
        plot the portfolio weights of last period? plot portfolio weights ob backtest etc..."""
        raise NotImplementedError()

    def ubah(self, flag):
        s_tr, e_tr, s_te, e_te = train_test_split(
            self.args.ratio,
            self.args.granularity,
            self.args.start_date,
            self.args.end_date,
        )
        start_date, end_date = (s_tr, e_tr) if flag == "train" else (s_te, e_te)
        start_date = add_periods_to_datetime(
            start_date, self.args.granularity, self.args.seq_len - 1
        )
        state_space = PriceHistory(
            self.args,
            num_periods=self.args.seq_len,
            granularity=self.args.granularity,
            start_date=start_date,
            end_date=end_date,
        )
        data = state_space.filled_feature_matrices[0]
        returns = calculate_returns(data).iloc[1:, :].values
        returns = np.add(returns, 1)
        returns_per_episode = returns.mean(axis=1)
        portfolio_values_ubah = [
            START_VALUE * np.prod(returns_per_episode[:i])
            for i in range(len(returns_per_episode))
        ]
        return portfolio_values_ubah[-1]
