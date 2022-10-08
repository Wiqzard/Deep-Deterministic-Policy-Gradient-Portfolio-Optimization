import os
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims: list[int, int, int],
        name,
        chkpt_dir="/content/ddpg",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims

        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        self.conv1 = nn.Conv2d(
            in_channels=self.input_dims[0], out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=20, kernel_size=(self.input_dims[1] - 2, 1)
        )
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))

        self.bn1 = nn.LayerNorm([1, 1, self.input_dims[2]])

        self.linear_state = nn.Linear(self.input_dims[2], 16)
        self.linear_action = nn.Linear(self.input_dims[2], 16)
        self.bn2 = nn.LayerNorm(16)
        self.linear_q = nn.Linear(16, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)

    def forward(self, state, action):
        # state_value = torch.tensor(state[0], dtype=torch.float32)
        state_value = state[0]
        state_value = nn.functional.relu(self.conv1(state_value))
        state_value = nn.functional.relu(self.conv2(state_value))
        state_value = torch.cat(
            (state_value, state[1].unsqueeze(0).unsqueeze(0)), dim=0
        )
        state_value = self.conv3(state_value)
        state_value = self.linear_state(state_value)

        action_value = nn.functional.relu(self.linear_action(action))

        state_action_value = nn.functional.relu(torch.add(state_value, action_value))
        state_action_value = self.linear_q(state_action_value)
        return state_action_value


# CNN
class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims: list[int, int, int],
        name,
        chkpt_dir="/content/ddpg",
    ):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        self.conv1 = nn.Conv2d(
            in_channels=self.input_dims[0], out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=20, kernel_size=(self.input_dims[1] - 2, 1)
        )
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)

    def forward(self, state):
        """
        input:       [num_features, num_periods, num_assets]
        conv1:    -> [2, num_periods-2, num_assets]
        conv2:    -> [20, 1, num_assets]  + w_last = [21, 1, num_assets]
        conv3:    -> [1, 1, num_assets] + cash_bias = [1, 1, num_assets + 1]
        softmax:  -> [num_assets + 1] normalized to 1
        """
        # state is tuple (X_t, w_t-1)
        w_t_1 = state[1].clone().detach().to(self.device)
        print(w_t_1.shape)
        x = state[0].clone().detach().to(self.device)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        print(x.shape)
        print(w_t_1.shape)
        x = torch.cat((x, w_t_1), dim=0)
        x = self.conv3(x).squeeze()
        # cash_bias = nn.Parameter(torch.zeros(1,1,1))
        # x = torch.cat((chash_bias, x), dim=2)
        x = nn.functional.softmax(x, dim=0)
        return x


class ActorNetworkLinear(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        fcl_dims,
        fc2_dims,
        n_actions,
        name,
        chkpt_dir="/content/ddpg",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc1.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)


class CriticNetworkLinaer(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        fcl_dims,
        fc2_dims,
        n_actions,
        name,
        chkpt_dir="/content/ddpg",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)
