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
        # r_t = math.log(mu_t * np.dot(y_t, w_t_1))
        r_t = mu_t * np.dot(y_t, w_t_1)
        rewards.append(r_t)
    return rewards


def calculate_rewards_torch(scales, states, prev_actions, actions, args):
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
        X_t = torch.tensor(X_t, dtype=torch.float32).float()
        w_t_1 = prev_actions[batch].float()
        if args.use_numeraire:
            y_t = X_t[args.seq_len, 1:] / X_t[args.seq_len - 1, 1:]
            y_t = torch.cat((np.ones((1)), y_t))
        else:
            y_t = X_t[args.seq_len - 1, :] / X_t[args.seq_len - 2, :]
        w_t_prime = (torch.multiply(y_t, w_t_1)) / torch.dot(y_t, w_t_1)
        mu_t = 1 - args.commission_rate_selling * sum(
            torch.abs(w_t_prime - actions[batch])
        )
        r_t = torch.log(mu_t * torch.dot(y_t, w_t_1))
        rewards.append(r_t)
    return rewards


def calculate_cummulative_reward(rewards):
    cumm_reward = []
    for i, element in enumerate(rewards):
        cumm_reward.append(element / (i + 1))
    print(cumm_reward)
    return sum(cumm_reward)


import math
import numpy as np
from tqdm import tqdm
from fed_former.data_factory import DataSet
import torch
from fed_former.agent_ts import Agent
from tests.test_args import args
from torch.utils.data import DataLoader

data = DataSet(args, flag="train")
agent = Agent(args, flag="train")
epoch = 2


data_loader = DataLoader(data, batch_size=3, shuffle=True, drop_last=True)


def train():
    for epoch_ in range(epoch):
        total_rewards = []
        for idxs, (states, prev_actions, next_states) in tqdm(
            data_loader, total=len(data_loader), leave=True
        ):
            actions = agent.choose_action(states, prev_actions, flag="train")
            actions = torch.tensor(actions, dtype=torch.float32)
            print(actions)
            data.action_memory.store_action(actions, idxs)
            rewards = calculate_rewards(states, prev_actions, actions, args)
            print(rewards)
            agent.learn(states, prev_actions, actions, next_states, rewards)
            total_rewards.append(rewards)


from fed_former.lstm import ActorLSTM
from fed_former.layers.embeddings import DataEmbedding
import torch.optim as optim

embedding = DataEmbedding(c_in=8, d_model=args.d_model, embed_type="timef", freq="t")
actor = ActorLSTM(args, embedding)

optimizer = optim.Adam(actor.parameters(), lr=args.actor_learning_rate)


def train2():
    for epoch_ in range(epoch):
        for idxs, scales, states, prev_actions, next_states in tqdm(
            data_loader, total=len(data_loader), leave=True
        ):
            states, _, state_time_marks, _ = states
            actions = actor(states, state_time_marks, prev_actions)
            rewards = calculate_rewards_torch(
                scales, states, prev_actions, actions, args
            )
            data.action_memory.store_action(actions.detach().numpy(), idxs)
            # reward = sum(rewards)
            reward = calculate_cummulative_reward(rewards)
            reward.backward()
            print(reward)
            optimizer.step()
            optimizer.zero_grad()


# train()
train2()
