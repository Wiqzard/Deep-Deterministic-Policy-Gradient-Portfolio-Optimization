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


def train2():
    for epoch_ in range(epoch):
        for idxs, scales, states, prev_actions, next_states in tqdm(
            data_loader, total=len(data_loader), leave=True
        ):
            actor.zero_grad()
            states, _, state_time_marks, _ = states
            print(states.shape)
            print(state_time_marks.shape)
            print(prev_actions.shape)
            actions = actor(states, state_time_marks, prev_actions)
            actions.requires_grad = True
            print(actions)
            rewards = calculate_rewards_torch(
                scales, states, prev_actions, actions, args
            )
            # data.action_memory.store_action(actions.detach().numpy(), idxs)
            # reward = sum(rewards)
            reward = calculate_cummulative_reward(rewards)
            # reward = criterion(actions, torch.zeros_like(actions))
            start = time.time()
            reward.backward()
            print(80 * "-")
            print(reward)
            optimizer.step()
            end = time.time()
            print("time", end - start)

            for name, param in actor.named_parameters():
                print(name, param)


# train()
class A:
    def __init__(self) -> None:
        self.commission_rate_selling = 0.0025

    def train3(self):
        actor = ActorLSTM(args, "timeF")
        actor.train()
        optimizer = optim.Adam(actor.parameters(), lr=0.1, maximize=True)
        print(actor.parameters())
        pytorch_total_params = sum(
            p.numel() for p in actor.parameters() if p.requires_grad
        )
        print(pytorch_total_params)
        criterion = self.reward
        for i in range(10):
            for j in range(100):
                optimizer.zero_grad()
                state = torch.randn((3, 50, 8))
                state_mark = torch.randn((3, 50, 5))
                prev_action = torch.randn((3, 8))
                next_state = torch.randn((3, 50, 8))
                scales = (torch.randn((3, 8)), torch.randn((3, 8)))
                self.previous_w = torch.abs(prev_action)
                self.future_price = torch.abs(
                    torch.divide(next_state[:, -1, :], next_state[:, -2, :])
                )
                actions = actor(state, state_mark, prev_action)

                reward_ = criterion(torch.abs(actions))
                # print(reward_)
                # rewards = calculate_rewards_torch(scales, state, prev_action, actions, args)
                # reward = calculate_cummulative_reward(rewards)
                # print(reward_)
                reward_.backward()
                print(actor.lstm.weight_ih_l0.grad)
                optimizer.step()

            for name, param in actor.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    def reward(self, actions):
        return torch.mean(
            torch.log(torch.sum(actions * self.future_price, dim=1))
            - torch.sum(
                torch.abs(actions - self.previous_w) * self.commission_rate_selling,
                dim=1,
            )
        )


# train3()
a = A()
a.train3()

#
# model = ActorLSTM(args, "timeF")
## print(check_backward_pass(model))
#
# print(list(model.parameters()))
#
# def reward(actions):
#    output = self.__net.output[:] #actions
#    future_price = self.__future_price #
#    output_sum = torch.sum(output * future_price, dim=1)
#    output_mean_log = -torch.mean(torch.log(output_sum))
#    prev_w = self.__net.previous_w
#    abs_output_diff = torch.abs(output[:, 1:] - prev_w)
#    commission_ratio = self.__commission_ratio
#    output_diff_mean = torch.mean(abs_output_diff * commission_ratio, dim=1)
#    return output_mean_log - output_diff_mean
#
