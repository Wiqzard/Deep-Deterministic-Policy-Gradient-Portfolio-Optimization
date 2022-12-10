from fed_former.data_factory import DataSet

from tests.test_args import args
from fed_former.agent_ts import Agent
import torch
import numpy as np
import math

data = DataSet(args, flag="train")

# print(len(data))
# print(data[0])
# print(data[0][1])
# print(data.action_memory.action_memory)
# print(data.sample_batch(3)[0][0])
# print(data.action_memory.store_actions())

from torch.utils.data import DataLoader

data_loader = DataLoader(data, batch_size=3, shuffle=True, drop_last=True)
print(next(iter(data_loader)))


def calculate_rewards(states, prev_actions, actions, args):
    """
    put that into data class
    args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
    -> Calculate y_t from X_t = v_t // v_t-1
    -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
    -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
    -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
    """
    seq_x_s = states[0]
    rewards = []
    for batch in range(seq_x_s.shape[0]):
        X_t = seq_x_s[batch]
        X_t = data.inverse_transform(X_t)
        w_t_1 = prev_actions[batch]
        if args.use_numeraire:
            y_t = X_t[args.seq_len, 1:] / X_t[args.seq_len - 1, 1:]
            y_t = np.concatenate((np.ones((1)), y_t))
        else:
            y_t = X_t[args.seq_len - 1, :] / X_t[args.seq_len - 2, :]

        w_t_prime = (np.multiply(y_t, w_t_1)) / np.dot(y_t, w_t_1)

        mu_t = 1 - args.commission_rate_selling * sum(
            np.abs(w_t_prime - actions[batch])
        )
        r_t = math.log(mu_t * np.dot(y_t, w_t_1))
        rewards.append(r_t)
    return rewards


agent = Agent(args, flag="train")

idxs, (states, prev_actions, next_states) = data.sample_batch(batch_size=3)

actions = agent.choose_action(states, prev_actions, flag="train")
actions = torch.tensor(actions, dtype=torch.float32)
print("actions")
print(actions.shape)
data.action_memory.store_action(actions, idxs)
# print(data.action_memory.action_memory)  # SEEMS NOT WORKING
# print(actions)
rewards = calculate_rewards(
    states, prev_actions, actions, args
)  # Not sure fi states or next states
# print(rewards)
print("pre learn")
print(prev_actions.shape)
print(actions.shape)
agent.learn(states, prev_actions, actions, next_states, rewards)
for itemn in data.action_memory.action_memory:
    print(itemn)
print(data.action_memory.action_memory)
