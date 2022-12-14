import logging
import argparse
import numpy as np
import random
import torch
import warnings
import os

from utils.constants import *
from data_management.coin_database import CoinDatabase
from exp.exp_main import Exp_Main
from utils.visualize import (
    plot_asset_values,
    plot_weights_last_backtest,
    plot_value_last_backtest,
    plot_results_episodes,
    plot_weight_changes_episodes,
)
from environment.environment import Environment


# agent
def main():
    warnings.filterwarnings("ignore")
    logger = logging.getLogger("__name__")
    level = logging.INFO
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="DDPG Portfolio Optimization")

    parser.add_argument("--is_training", action="store_true", help="status")

    parser.add_argument(
        "--colab",
        action="store_true",
        default=False,
        help="has to be set if used in google colab",
    )
    parser.add_argument(
        "--reward_multiplier", type=int, default=1, help="increase reward for learning"
    )
    parser.add_argument("--conv1_out", type=int, default=32, help="32 The output size of conv1")
    parser.add_argument("--conv2_out", type=int, default=32, help=" 64The output size of conv2")
    parser.add_argument("--conv3_out", type=int, default=16, help="32 The output size of conv3")
    parser.add_argument("--fc1_out", type=int, default=64, help="64 The output size of fc1")
    parser.add_argument(
        "--use_numeraire",
        type=bool,
        default=True,
        help="Use the first coin as riskless asset.",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="OU",
        help="type of noise to use for the DDPG agent",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="sigma parameter for the Ornstein-Uhlenbeck noise",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.25,
        help="theta parameter for the Ornstein-Uhlenbeck noise",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="time step for the Ornstein-Uhlenbeck noise",
    )
    parser.add_argument(
        "--x0",
        type=int,
        default=None,
        help="initial value for the Ornstein-Uhlenbeck noise",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for training the DDPG agent",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for the DDPG agent"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1e-2,
        help="soft update parameter for the DDPG agent",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=100000,
        help="maximum size of the replay buffer for the DDPG agent",
    )
    parser.add_argument(
        "--scalar",
        type=float,
        default=0.1,
        help="scalar for random weights",
    )
    parser.add_argument(
        "--scalar_decay",
        type=float,
        default=0.992,
        help="decay of scalar duhhh",
    )
    parser.add_argument(
        "--desired_distance",
        type=float,
        default=0.1,
        help="no idea what it does",
    )
    # Networks
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=1e-4,
        help="learning rate for the critic network",
    )
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for the actor network",
    )
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default="outputs/ddpg",
        help="directory to save the checkpoints",
    )

    # training
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="number of episodes to train the DDPG agent",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="ratio of training to testing data"
    )
    parser.add_argument(
        "--benchmark_name", type=str, default="UBAH", help="name of the benchmark model"
    )
    parser.add_argument(
        "--compute_before",
        action="store_true",
        default=False,
        help="flag to indicate whether to compute the benchmark before training",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="sequence length for the data used by the DDPG agent",
    )

    # database stuff
    parser.add_argument(
        "--database_path",
        type=str,
        default="outputs/coin_history.db",
        help="path to the database containing the historical data",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=900,
        help="granularity of the historical data in seconds",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2022-05-10-00-00",
        help="start date of the historical data in YYYY-MM-DD-HH-MM format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="end date of the historical data in YYYY-MM-DD-HH-MM format",
    )

    parser.add_argument(
        "--commission_rate_selling",
        type=float,
        default=0.0025,
        help="commission rate for selling",
    )
    parser.add_argument(
        "--commission_rate_purchasing",
        type=float,
        default=0.0025,
        help="commission rate for purchasing",
    )

    parser.add_argument(
        "--fill_database",
        action="store_true",
        default=False,
        help="flag to indicate whether to fill the database with historical data",
    )

    parser.add_argument("--use_gpu", action="store_true", default=True, help="use gpu")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    parser.add_argument(
        "--with_test", default=True, action="store_true", help="test in training"
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume training from the last checkpoint",
    )

    args = parser.parse_args()

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    # <-----------------------= START RUN =-----------------------> #

    logger.info("Args in experiment:")
    logger.info(args)

    if args.fill_database:
        data_base = CoinDatabase(args)
        data_base.create_all_tables()
        data_base.fill_all_tables(
            granularity=args.granularity,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    Exp = Exp_Main
    if args.is_training:
        exp = Exp(args)
        setting = f"setting: alr_{args.actor_learning_rate}, clr_{args.critic_learning_rate}, mem_, ..."
        logger.info("\n >>>>>>> start training : --- >>>>>>>>>>>>>>>>>>>>>>>>>> \n ")

        exp.train(args.with_test, args.resume)
    from data_management.data_manager import PriceHistory

    price_history = PriceHistory(
        args, args.seq_len, args.granularity, args.start_date, args.end_date
    )
    prices = price_history.filled_feature_matrices[0]
    plot_asset_values(
        prices, args.granularity, scale=True, difference=False, save_path=None
    )
    #  train_scores_episodes, test_scores_episodes, train_action_histories, test_action_histories = Exp.get_results
    plot_results = False
    if plot_results:
        path_results = "outputs/results"
        if not os.listdir(path_results):
            logger.warn("The path is empty")
        else:
            train_scores_episodes = np.load(
                os.path.join(path_results, "train_scores_episodes.npy"),
                allow_pickle=True,
            )
            test_scores_episodes = np.load(
                os.path.join(path_results, "test_scores_episodes.npy"),
                allow_pickle=True,
            )
            train_action_histories = np.load(
                os.path.join(path_results, "train_action_histories.npy"),
                allow_pickle=True,
            )
            test_action_histories = np.load(
                os.path.join(path_results, "test_action_histories.npy"),
                allow_pickle=True,
            )
            plot_train = True
            print(train_scores_episodes, test_scores_episodes)
            last_train_action_history = train_action_histories[-1]
            last_test_action_history = test_action_histories[-1]
            last_train_scores = train_scores_episodes[-1]
            last_test_scores = test_scores_episodes[-1]

            if plot_train:
                plot_weights_last_backtest(last_train_action_history, k=1)
                plot_value_last_backtest(last_train_scores, k=1)
                plot_results_episodes(train_scores_episodes, k=1)
                plot_weight_changes_episodes(train_action_histories, k=1)
            else:
                plot_weights_last_backtest(last_test_action_history, k=1)
                plot_value_last_backtest(last_test_scores, k=1)
                plot_results_episodes(test_scores_episodes, k=1)
                plot_weight_changes_episodes(test_action_histories, k=1)
    #


#    #PLOT ASSET VALUES
#   env = Environment(args)
#    plot_asset_values(env, scale=True, difference=False)


if __name__ == "__main__":
    main()
