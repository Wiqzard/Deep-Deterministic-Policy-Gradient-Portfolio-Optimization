import logging
import argparse
import numpy as np
import random
import torch
import warnings

from utils.constants import *
from data_management.coin_database import CoinDatabase
from utils.tools import logger
from exp.exp_main import Exp_Main

#agent
def main():
    warnings.filterwarnings('ignore')

    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="DDPG Portfolio Optimization")

    parser.add_argument(
        "--is_training", action="store_true", help="status"
    )

    parser.add_argument("--noise", type=str, default="OU", help="type of noise to use for the DDPG agent")
    parser.add_argument("--sigma", type=float, default=0.15, help="sigma parameter for the Ornstein-Uhlenbeck noise")
    parser.add_argument("--theta", type=float, default=0.15, help="theta parameter for the Ornstein-Uhlenbeck noise")
    parser.add_argument("--dt", type=float, default=0.002, help="time step for the Ornstein-Uhlenbeck noise")
    parser.add_argument("--x0", type=int, default=None, help="initial value for the Ornstein-Uhlenbeck noise")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training the DDPG agent")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for the DDPG agent")
    parser.add_argument("--tau", type=float, default=1e-3, help="soft update parameter for the DDPG agent")
    parser.add_argument("--max_size", type=int, default=100000, help="maximum size of the replay buffer for the DDPG agent")

    # Networks
    parser.add_argument("--critic_learning_rate", type=float, default=1e-4, help="learning rate for the critic network")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-3, help="learning rate for the actor network")
    parser.add_argument("--chkpt_dir", type=str, default="outputs/ddpg", help="directory to save the checkpoints")

    # training
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes to train the DDPG agent")
    parser.add_argument("--ratio", type=float, default=0.8, help="ratio of training to testing data")
    parser.add_argument("--benchmark_name", type=str, default="UBAH", help="name of the benchmark model")
    parser.add_argument("--compute_before", action="store_true", default=False, help="flag to indicate whether to compute the benchmark before training")
    parser.add_argument("--seq_len", type=int, default=50, help="sequence length for the data used by the DDPG agent")

    # database stuff
    parser.add_argument("--database_path", type=str, default="coin_history.db", help="path to the database containing the historical data")
    parser.add_argument("--granularity", type=int, default=900, help="granularity of the historical data in seconds")
    parser.add_argument("--start_date", type=str, default="2022-05-10-00-00", help="start date of the historical data in YYYY-MM-DD-HH-MM format")
    parser.add_argument("--end_date", type=str, default=None, help="end date of the historical data in YYYY-MM-DD-HH-MM format")

    parser.add_argument("--commission_rate_selling", type=float, default=0.0025, help="commission rate for selling")
    parser.add_argument("--commission_rate_purchasing", type=float, default=0.0025, help="commission rate for purchasing")

    parser.add_argument("--fill_database", action="store_true", default=False, help="flag to indicate whether to fill the database with historical data")

    parser.add_argument("--use_gpu", action="store_true", default=True, help="use gpu")
    parser.add_argument("--use_amp",action="store_true",help="use automatic mixed precision training",default=False)

    parser.add_argument("--with_test", default=True, action="store_true",
                    help="test in training")
    parser.add_argument("--resume", default=False, action="store_true",
                    help="Resume training from the last checkpoint")

    args = parser.parse_args()

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    logger.info("Args in experiment:")
    logger.info(args)


    if args.fill_database:
        data_base = CoinDatabase(args)
        data_base.create_all_tables()
        data_base.fill_all_tables(
        granularity=args.granularity, start_date=args.start_date, end_date=args.end_date
        )

    Exp = Exp_Main
    if args.is_training:
        exp = Exp(args)
        logger.info("\n >>>>>>> start training : --- >>>>>>>>>>>>>>>>>>>>>>>>>> \n ")
        
        exp.train(args.with_test, args.resume)

    from utils.visualize import plot_asset_values
    from environment.environment import Environment
    env = Environment(args)
    plot_asset_values(env, scale=True, difference=False)
if __name__ == "__main__":
    main()