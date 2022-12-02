import logging
import argparse
import numpy as np
import random
import torch

from utils.constants import *
from data_management.coin_database import CoinDatabase


if args.fill_database:
    data_base = CoinDatabase(args)
    data_base.create_all_tables()
    data_base.fill_all_tables(
      granularity=args.granularity, start_date=args.start_date, end_date=args.end_date
    )

args.seq_len = 50
#database stuff
args.database_path = "coin_history.db"
args.granularity = 900
args.start_date = "2022-05-10-00-00"
args.end_date = None

args.commission_rate_selling = 0.0025
args.commission_rate_purchasing = 0.0025

def main():
    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="DDPG Portfolio Optimization")

    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )


    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    args = parser.parse_args()

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    logger.info("Args in experiment:")
    logger.info(args)


if __name__ == "__main__":
    main()