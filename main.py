import torch
import numpy as np
import os


from utils.tools import dotdict
from utils.tools import logger, add_periods_to_datetime


args = dotdict()
args.is_training = True
args.colab = True
args.linear2 = 64
args.conv_dim = 64
args.noise = "OU"
args.sigma = 0  # 0.25
args.theta = 0.25
args.dt = 0.002
args.x0 = None
args.batch_size = 64
args.gamma = 0.99
args.tau = 1e-2
args.max_size = 100000

args.conv1_out = 32  # 32
args.conv2_out = 32  # 64
args.conv3_out = 16  # 32
args.fc1_out = 64  # 128


args.critic_learning_rate = 1e-4
args.actor_learning_rate = 1e-3
args.chkpt_dir = "outputs/ddpg"

args.episodes = 500
args.ratio = 0.8
args.benchmark_name = "UBAH"
args.compute_before = False
args.seq_len = 50
args.reward_multiplier = 1
args.database_path = "outputs/coin_history.db"
args.granularity = 900
args.start_date = "2022-10-09-00-00"
args.end_date = "2022-10-20-00-00"

args.commission_rate_selling = 0.00
args.commission_rate_purchasing = 0.0025

args.fill_database = True
args.with_test = True
args.resume = False

args.use_gpu = True
args.use_amp = False
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
logger.info("Args in experiment:")
logger.info(args)

from utils.visualize import (
    plot_asset_values,
    plot_weights_last_backtest,
    plot_value_last_backtest,
    plot_results_episodes,
    plot_weight_changes_episodes,
)

# model_names = ["CRP", "UBAH", "BCRP", "BestMarkowitz", "UP", "Anticor", "OLMAR", "RMR"]
# plot_model(args, model_name="OLMAR")


from data_management.data_manager import PriceHistory
from utils.tools import train_test_split
from environment.environment import Environment


from fed_former.data_factory import DataSet

from exp.exp_main import Exp_Main

exp = Exp_Main(args)
# exp.agent.critic.load_checkpoint()
# exp.agent.actor.load_checkpoint()
# exp.train(resume=False)
exp.backtest()  #
# obs, _ = env = exp.train_env.reset()
# act = exp.agent.choose_action(obs)
# print(act)
# new_state, reward, done = exp.train_env.step(act)
# act2 = exp.agent.choose_action(new_state)
# print(act2)
# print(exp.test_env.state_history)
# print(exp.test_env.action_history)
# data = DataSet(args, "train")
# print(data.scaled_close_prices)
# x = torch.tensor(data.scaled_close_prices.astype(np.float)).float().unsqueeze(0)
## torch.Size([1, 4441])
# x_mark = torch.tensor(data.data_stamp).float().unsqueeze(0)
## torch.Size([1, 4441, 5])
# inp = embed(x, x_mark)
# print(inp)
# print(inp.shape)
## torch.Size([1, 4441, 512])
# an = sum(p.numel() for p in embed.parameters() if p.requires_grad)
# print(an)
#
#
# import torch.nn as nn
# import torch.functional as F
# import torch.optim as optim
# args.d_model = 512
# args.hidden_size = 256
# args.num_layers = 4
# args.fc1_out = 128
# args.fc2_out = 32
# args.fc3_out = 16
#
# from fed_former.lstm import ActorLSTM, CriticLSTM
# actor = ActorLSTM(args)
# critic = CriticLSTM(args)
#
# hid = torch.randn((1, 50, 16))
# lstm = LSTM(16, 4)
# out, hidden = lstm(inp[:, :50, :], hid)
# print(out)
# def test_steps():
#    prices = PriceHistory(
#        args,
#        num_periods=args.seq_len,
#        granularity=args.granularity,
#        start_date=args.start_date,
#        end_date=args.end_date,
#    )
#    start_date_train, end_date_train, start_date_test, end_date_test = train_test_split(
#        args.ratio, args.granularity, args.start_date, args.end_date
#    )
#    print(f"start_date_train: {start_date_train}")
#    print(f"end_date_train: {end_date_train}")
#    print(f"start_date_test: {start_date_test}")
#    print(f"end_date_test: {end_date_test}")
#    prices_train = PriceHistory(
#        args,
#        num_periods=args.seq_len,
#        granularity=args.granularity,
#        start_date=start_date_train,
#        end_date=end_date_train,
#    )
#    prices_test = PriceHistory(
#        args,
#        num_periods=args.seq_len,
#        granularity=args.granularity,
#        start_date=start_date_test,
#        end_date=end_date_test,
#    )
#    print(f"length of prices: {len(prices)}")
#    print(f"length of prices_train: {len(prices_train)}")
#    print(f"length of prices_test: {len(prices_test)}")
#
#    env = Environment(args=args, flag="full")
#    env_train = Environment(args=args, flag="train")
#    env_test = Environment(args=args, flag="test")
#
#    print(env.get_numeraire_ratio())
#    print(env.state_space.filled_feature_matrices[0])
#
#    print(f"number of steps in env: {env.num_steps}")
#    print(f"number of steps in env_train: {env_train.num_steps}")
#    print(f"number of steps in env_test: {env_test.num_steps}")
#
#    start_date_test = add_periods_to_datetime(
#        start_date_test, args.granularity, args.seq_len
#    )
#    p = PriceHistory(
#        args,
#        num_periods=args.seq_len,
#        granularity=args.granularity,
#        start_date=start_date_test,
#        end_date=end_date_test,
#    )
#    plot_asset_values(p.filled_feature_matrices[0], args.granularity, True)
#
#
# def plot_test():
#    path_results = "outputs/results"
#    if not os.listdir(path_results):
#        logger.warn("The path is empty")
#    else:
#        train_scores_episodes = np.load(
#            os.path.join(path_results, "train_scores_episodes.npy"), allow_pickle=True
#        )
#        test_scores_episodes = np.load(
#            os.path.join(path_results, "test_scores_episodes.npy"), allow_pickle=True
#        )
#        train_action_histories = np.load(
#            os.path.join(path_results, "train_action_histories.npy"), allow_pickle=True
#        )
#        test_action_histories = np.load(
#            os.path.join(path_results, "test_action_histories.npy"), allow_pickle=True
#        )
#        plot_train = False
#        last_train_action_history = train_action_histories[-1]
#        last_test_action_history = test_action_histories[-1]
#        last_train_scores = train_scores_episodes[-1]
#        last_test_scores = test_scores_episodes[-1]
#
#        if plot_train:
#            plot_weights_last_backtest(last_train_action_history, k=1)
#            plot_value_last_backtest(last_train_scores, args, "train", k=1)
#            plot_results_episodes(train_scores_episodes)
#            plot_weight_changes_episodes(train_action_histories)
#        else:
#            plot_weights_last_backtest(last_test_action_history, k=1)
#            plot_value_last_backtest(last_test_scores, args, "test", k=1)
#            plot_results_episodes(test_scores_episodes)
#            plot_weight_changes_episodes(test_action_histories)
#
#
## test_steps()
## plot_test()
#
# from utils.visualize import plot_portfolio_algos, plot_model
#
# args.commission_rate_selling = 0
## plot_model(args, "OLMAR", flag="full")
## plot_portfolio_algos(args, flag="full")
## plot_portfolio_algos(args, "full")
#
# from agent.networks import ActorNetwork, CriticNetwork
#
# actor = ActorNetwork(args)
# critic = CriticNetwork(args)
#
# an = sum(p.numel() for p in actor.parameters() if p.requires_grad)
# cn = sum(p.numel() for p in critic.parameters() if p.requires_grad)
#
# print(an)
# print(cn)
#
# from agent.agent import Agent
#
# env = Environment(args, flag="train")
# obs, _ = env.reset()
# agent = Agent(args, flag="train")
# for _ in range(0, 10):
#    act = agent.choose_action(obs)
#    print(act)
#    state, r, d = env.step(act)
#    obs = state
#
