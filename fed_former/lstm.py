import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from utils.constants import *
from utils.tools import logger
from fed_former.layers.embeddings import DataEmbedding


class ActorLSTM(nn.Module):
    def __init__(self, args, embed_type="timef", freq="t"):
        super(ActorLSTM, self).__init__()
        self.args = args
        self.create_checkpoint(name="actor")
        self.embedding = DataEmbedding(
            NUM_ASSETS, d_model=args.d_model, embed_type=embed_type, freq=freq
        )  # embedding  # DataEmbedding

        self.lstm = nn.LSTM(
            input_size=args.d_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(args.hidden_size * args.seq_len, args.fc1_out)
        self.fc2 = nn.Linear(args.fc1_out, args.fc2_out)
        self.fc3 = nn.Linear(args.fc2_out + NUM_ASSETS, NUM_ASSETS)

        self.drop_layer = nn.Dropout(p=args.dropout)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float().to(self.device)

    def create_checkpoint(self, name):
        self.name = name
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_dpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file missing.")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def init_hidden(self, seq_len):
        return (
            torch.zeros(self.args.num_layers, seq_len, self.args.d_model),
            torch.zeros(self.args.num_layers, seq_len, self.args.d_model),
        )

    def forward(self, state, time_mark, prev_action, hidden=None):
        action_w_1 = prev_action.to(self.device)
        state_value = state.to(self.device)
        time_mark = time_mark.to(self.device)
        # if not hidden:
        #   hidden = self.init_hidden(time_mark.shape[-1])
        embed = self.embedding(state_value, time_mark)
        output, (hidden_state, cell_state) = self.lstm(embed)  # , hidden)
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        action = F.leaky_relu(self.fc1(output))
        action = torch.flatten(action, start_dim=1, end_dim=-1)
        action = F.leaky_relu(self.drop_layer(self.fc2(action)))
        action = torch.cat((action, action_w_1), dim=-1)
        action = self.fc3(action).squeeze()
        action = action / torch.norm(action, p=2, dim=-1, keepdim=True)
        if self.args.bb:
            print(action)
        action = F.softmax(action, dim=-1)
        if self.args.ab:
            print(action)
        return action

    def create_checkpoint(self, name):
        chkpt_dir = self.args.chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

    def save_checkpoint(self):
        if not self.checkpoint_file:
            raise ValueError("Checkpoint file missing.")
        # print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticLSTM(nn.Module):
    def __init__(self, args, embedding):
        super(CriticLSTM, self).__init__()
        self.args = args
        self.dropout = 0.1
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=args.d_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(args.hidden_size, args.fc1_out)
        self.fc2 = nn.Linear(args.fc1_out, args.fc2_out)
        self.fc3 = nn.Linear(args.fc2_out + NUM_ASSETS, args.fc3_out)
        self.fc4 = nn.Linear(args.fc3_out + NUM_ASSETS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float().to(self.device)

    def forward(self, state, time_mark, prev_action, action, hidden=None):
        action_w_1 = prev_action.to(self.device)
        action = action.to(self.device)
        state_value = state.to(self.device)
        time_mark = time_mark.to(self.device)

        embed = self.embedding(state_value, time_mark)

        if not hidden:
            self.hidden = self.init_hidden(time_mark.shape[-1])
        output, (hidden_state, cell_state) = self.lstm(embed)
        hidden_state = hidden_state.permute(1, 0, 2).squeeze(1)
        state = F.relu(self.fc1(hidden_state))
        state = F.relu(self.fc2(state))

        state = torch.cat((state, action_w_1), dim=-1)
        state = F.relu(self.fc3(state))

        state_action = torch.cat((state, action), dim=-1)
        state_action_value = self.fc4(state_action).squeeze()
        return state_action_value

    def init_hidden(self, seq_len):
        return (
            torch.zeros(self.args.num_layers, seq_len, self.args.d_model),
            torch.zeros(self.args.num_layers, seq_len, self.args.d_model),
        )

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


class ActorNetwork2(nn.Module):
    def __init__(self, args):
        super(ActorNetwork2, self).__init__()
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

    def forward(self, state, prev_action):
        """
        input:      ([batch, num_features, num_periods, num_assets], [batch, 1 ,1 , num_assets])
        conv1:    -> [2, num_periods-2, num_assets] (2, 48, 8)
        conv2:    -> [20, 1, num_assets]  + w_last = [21, 1, num_assets]
        conv3:    -> [1, 1, num_assets] + cash_bias = [1, 1, num_assets (+ 1)]
        softmax:  -> [num_assets + 1] normalized to 1
        """
        action_w_1 = prev_action.to(self.device)
        state_value = state.to(self.device)
        print(state_value.shape)
        x = F.leaky_relu(self.conv1(state_value))
        print(x.shape)
        x = F.leaky_relu(self.conv2(x)).squeeze(-2).permute(0, 2, 1)
        print(x.shape)
        x = torch.cat((x, action_w_1), dim=-1).permute(0, 2, 1).unsqueeze(-1)
        action = self.conv3(x).squeeze()
        action = F.leaky_relu(self.linear(action))
        action = F.leaky_relu(self.linear2(action))
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
        action = F.softmax(action)
        return action
