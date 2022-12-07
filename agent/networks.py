import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from utils.constants import *
from utils.tools import logger


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=args.conv1_out, kernel_size=(5, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=args.conv1_out,
            out_channels=args.conv2_out,
            kernel_size=(5, 3),
        )
        self.conv3 = nn.Conv2d(
            in_channels=args.conv2_out, out_channels=args.conv3_out, kernel_size=(5, 2)
        )
        self.layer_norm = nn.LayerNorm((args.conv3_out * 17 * 5))
        self.fc1 = nn.Linear(
            in_features=args.conv3_out * 17 * 5, out_features=args.fc1_out
        )
        self.fc2 = nn.Linear(in_features=args.fc1_out, out_features=32)
        self.fc3 = nn.Linear(in_features=40, out_features=16)
        self.fc4 = nn.Linear(in_features=24, out_features=1)

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float().to(self.device)

    def forward(self, state, action):
        action_1 = state[1].to(self.device)
        action = action.to(self.device)
        state_value = state[0].to(self.device)

        state_value = F.relu(self.conv1(state_value))
        state_value = F.relu(self.conv2(state_value))
        state_value = F.max_pool2d(state_value, (2, 1))
        state_value = self.conv3(state_value)

        state_value = torch.flatten(state_value, 1)
        state_value = self.layer_norm(state_value)
        state_value = F.relu(self.fc1(state_value))
        state_value = F.sigmoid(self.fc2(state_value))

        state_value = torch.cat((state_value, action_1), dim=-1)
        state_value = F.relu(self.fc3(state_value))

        state_action_value = torch.cat((state_value, action), dim=-1)
        state_action_value = self.fc4(state_action_value)
        print(state_action_value.squeeze())
        return state_action_value.squeeze()

    def create_checkpoint(self, name):
        self.name = name
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file missing.")
        print(f"... saving checkpoint ... {self.name}")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f"... loading checkpoint ... {self.name}")
        self.load_state_dict(torch.load(self.checkpoint_file))


# CNN
class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=args.conv1_out, kernel_size=(5, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=args.conv1_out,
            out_channels=args.conv2_out,
            kernel_size=(5, 3),
        )

        self.conv3 = nn.Conv2d(
            in_channels=args.conv2_out, out_channels=args.conv3_out, kernel_size=(5, 2)
        )
        self.fc1 = nn.Linear(
            in_features=args.conv3_out * 17 * 5, out_features=args.fc1_out
        )

        self.layer_norm = nn.LayerNorm((args.conv3_out * 17 * 5))

        self.fc2 = nn.Linear(in_features=args.fc1_out, out_features=32)
        self.fc3 = nn.Linear(in_features=40, out_features=8)
        self.batch_norm_layer = nn.BatchNorm1d(NUM_ASSETS)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float().to(self.device)

    def create_checkpoint(self, name):
        self.name = name
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file missing.")
        print(f"... saving checkpoint ... {self.name}")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f"... loading checkpoint ... {self.name}")
        self.load_state_dict(torch.load(self.checkpoint_file))

    def add_parameter_noise(self, scalar=None):
        scalar = scalar or self.args.scalar
        self.conv1.weight.data += torch.randn_like(self.conv1.weight.data) * scalar
        self.conv3.weight.data += torch.randn_like(self.conv3.weight.data) * scalar
        self.linear.weight.data += torch.randn_like(self.linear.weight.data) * scalar

    def forward(self, state):
        """
        input:      ([batch, num_features, num_periods, num_assets], [batch, 1 ,1 , num_assets])
        conv1:    -> [2, num_periods-2, num_assets] (2, 48, 8)
        conv2:    -> [20, 1, num_assets]  + w_last = [21, 1, num_assets]
        conv3:    -> [1, 1, num_assets] + cash_bias = [1, 1, num_assets (+ 1)]
        softmax:  -> [num_assets + 1] normalized to 1
        """
        action_1 = state[1].to(self.device)
        state_value = state[0].to(self.device)

        x = F.relu(self.conv1(state_value))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 1))
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))

        action = F.sigmoid(self.fc2(x))
        action = torch.cat((action, action_1), dim=-1)
        action = self.fc3(action)
        #        cash_bias = torch.cat(
        #            (
        #                torch.ones(*action.shape[:-1], 1),
        #                torch.zeros(*action.shape[:-1], action.shape[-1] - 1),
        #            ),
        #            dim=-1,
        #        ).to(self.device)
        #        if self.args.use_numeraire:
        #            action = torch.add(action, cash_bias)
        if self.args.sigm:
            action = action.squeeze()
            action = torch.add(F.sigmoid(action), -0.5)
        else:
            action = self.batch_norm_layer(action).squeeze()
        if self.args.bb:
            print("nn")
            print(action)
        if self.args.ab:
            if action.shape[0] < 5:
                print(action)
        return action


class CriticNetwork2(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=args.conv_dim, kernel_size=(args.seq_len - 2, 1)
        )
        self.conv3 = nn.Conv2d(
            in_channels=args.conv_dim + 1, out_channels=1, kernel_size=(1, 1)
        )

        self.bn1 = nn.LayerNorm([1, 1, NUM_ASSETS])

        self.linear_state1 = nn.Linear(NUM_ASSETS, 16)
        # self.linear_state2 = nn.Linear(16, 16)

        self.linear_action = nn.Linear(NUM_ASSETS, 16)
        self.bn2 = nn.LayerNorm(16)
        self.linear_q = nn.Linear(16, 1)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def create_checkpoint(self, name):
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file is missing.")
        logger.info("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logger.info("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))

    def forward(self, state, action):
        action_1 = state[1].unsqueeze(-2).to(self.device)  # torch.Size([10, 1, 8])
        action = action.to(self.device)
        state_value = state[0].to(self.device)  # torch.Size([10, 3, 50, 8])
        state_value = self.relu(self.conv1(state_value))  # torch.Size([10, 2, 48, 8])
        state_value = self.relu(self.conv2(state_value)).squeeze(
            -2
        )  # torch.Size([10, 20,  8])
        state_value = torch.cat((state_value, action_1), dim=1).unsqueeze(
            -1
        )  # torch.Size([10, 21, 8, 1])
        state_value = self.conv3(state_value).squeeze()  # torch.Size([10, 8])
        state_value = self.linear_state1(state_value)
        # state_value = self.linear_state2(state_value)
        action_value = self.linear_action(action)
        state_action_value = self.relu(torch.add(state_value, action_value))
        state_action_value = self.linear_q(state_action_value)
        # print(state_action_value)
        return state_action_value.squeeze()


class ActorNetwork2(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=args.conv_dim, kernel_size=(args.seq_len - 2, 1)
        )  # 64 20

        self.conv3 = nn.Conv2d(
            in_channels=args.conv_dim + 1, out_channels=1, kernel_size=(1, 1)
        )
        self.linear = nn.Linear(in_features=NUM_ASSETS, out_features=args.linear2)
        self.linear2 = nn.Linear(in_features=args.linear2, out_features=NUM_ASSETS)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def create_checkpoint(self, name):
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file missing.")
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))

    def add_parameter_noise(self, scalar=None):
        scalar = scalar or self.args.scalar
        self.conv1.weight.data += torch.randn_like(self.conv1.weight.data) * scalar
        self.conv3.weight.data += torch.randn_like(self.conv3.weight.data) * scalar
        self.linear.weight.data += torch.randn_like(self.linear.weight.data) * scalar

    def forward(self, state):
        """
        input:      ([batch, num_features, num_periods, num_assets], [batch, 1 ,1 , num_assets])
        conv1:    -> [2, num_periods-2, num_assets] (2, 48, 8)
        conv2:    -> [20, 1, num_assets]  + w_last = [21, 1, num_assets]
        conv3:    -> [1, 1, num_assets] + cash_bias = [1, 1, num_assets (+ 1)]
        softmax:  -> [num_assets + 1] normalized to 1
        """
        action_1 = state[1].unsqueeze(-1).to(self.device)
        state_value = state[0].to(self.device)

        x = self.relu(self.conv1(state_value))
        x = self.relu(self.conv2(x)).squeeze(-2).permute(0, 2, 1)
        x = torch.cat((x, action_1), dim=-1).permute(0, 2, 1).unsqueeze(-1)
        action = self.conv3(x).squeeze()
        action = self.relu(self.linear(action))
        action = self.relu(self.linear2(action))
        cash_bias = torch.cat(
            (
                torch.ones(*action.shape[:-1], 1),
                torch.zeros(*action.shape[:-1], action.shape[-1] - 1),
            ),
            dim=-1,
        ).to(self.device)
        if self.args.use_numeraire:
            action = torch.add(action, cash_bias)
        # print(action)
        action = self.softmax(action)
        return action
