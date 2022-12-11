from utils.tools import dotdict
from exp.exp_fed import Exp_Fed

args = dotdict()
args.colab = False
args.bb = False  # True
args.ba = False  # True
args.ab = False  # True

args.episodes = 500
args.ratio = 0.8
args.benchmark_name = "UBAH"
args.compute_before = False
args.seq_len = 50

args.database_path = "outputs/coin_history.db"
args.granularity = 900
args.start_date = "2022-10-01-00-00"
args.end_date = "2022-10-20-00-00"
args.commission_rate_selling = 0.0025
args.commission_rate_purchasing = 0.0025

# args.chkpt_dir = "contents/outputs/dpg"
args.chkpt_dir = "outputs/dpg"
args.d_model = 512  # 64
args.embed_type = "timef"
args.hidden_size = 256  # 64
args.num_layers = 1
args.fc1_out = 16
args.fc2_out = 16
args.dropout = 0.1

args.optim = "adam"
args.actor_learning_rate = 1e-3

args.batch_size = 32
args.shuffle = False
args.drop_last = False
args.num_workers = 0

args.use_gpu = False
args.use_amp = False

exp = Exp_Fed(args=args)
# exp.train(with_test=True)
dataloader = exp.get_dataloader(flag="train")

print(next(iter(dataloader)))
# from fed_former.data_factory import DataSet
#
# from tests.test_args import args
# from fed_former.agent_ts import Agent
# import torch
# import numpy as np
# import math
#
# data = DataSet(args, flag="train")
#
## print(len(data))
## print(data[0])
## print(data[0][1])
## print(data.action_memory.action_memory)
## print(data.sample_batch(3)[0][0])
## print(data.action_memory.store_actions())
#
# from torch.utils.data import DataLoader
#
# data_loader = DataLoader(data, batch_size=3, shuffle=True, drop_last=True)
# print(next(iter(data_loader)))
#
#
# def calculate_rewards(states, prev_actions, actions, args):
#    """
#    put that into data class
#    args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
#    -> Calculate y_t from X_t = v_t // v_t-1
#    -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
#    -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
#    -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
#    """
#    seq_x_s = states[0]
#    rewards = []
#    for batch in range(seq_x_s.shape[0]):
#        X_t = seq_x_s[batch]
#        X_t = data.inverse_transform(X_t)
#        w_t_1 = prev_actions[batch]
#        if args.use_numeraire:
#            y_t = X_t[args.seq_len, 1:] / X_t[args.seq_len - 1, 1:]
#            y_t = np.concatenate((np.ones((1)), y_t))
#        else:
#            y_t = X_t[args.seq_len - 1, :] / X_t[args.seq_len - 2, :]
#
#        w_t_prime = (np.multiply(y_t, w_t_1)) / np.dot(y_t, w_t_1)
#
#        mu_t = 1 - args.commission_rate_selling * sum(
#            np.abs(w_t_prime - actions[batch])
#        )
#        r_t = math.log(mu_t * np.dot(y_t, w_t_1))
#        rewards.append(r_t)
#    return rewards
#
#
# agent = Agent(args, flag="train")
#
# idxs, (states, prev_actions, next_states) = data.sample_batch(batch_size=3)
#
# actions = agent.choose_action(states, prev_actions, flag="train")
# actions = torch.tensor(actions, dtype=torch.float32)
# print("actions")
# print(actions.shape)
# data.action_memory.store_action(actions, idxs)
## print(data.action_memory.action_memory)  # SEEMS NOT WORKING
## print(actions)
# rewards = calculate_rewards(
#    states, prev_actions, actions, args
# )  # Not sure fi states or next states
## print(rewards)
# print("pre learn")
# print(prev_actions.shape)
# print(actions.shape)
# agent.learn(states, prev_actions, actions, next_states, rewards)
# for itemn in data.action_memory.action_memory:
#    print(itemn)
# print(data.action_memory.action_memory)
#
