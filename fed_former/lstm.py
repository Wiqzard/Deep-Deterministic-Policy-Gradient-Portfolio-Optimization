import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from utils.constants import *
from utils.tools import logger
from fed_former.layers.embeddings import DataEmbedding


class ActorLSTM(nn.Module):
    def __init__(self, args, embedding):
        super(ActorLSTM, self).__init__()
        self.args = args
        self.timeenc = "timef"
        self.dropout = 0.1
        self.embedding = embedding  # DataEmbedding

        self.lstm = nn.LSTM(
            input_size=args.d_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(args.hidden_size, args.fc1_out)
        self.fc2 = nn.Linear(args.fc1_out, args.fc2_out)
        self.fc3 = nn.Linear(args.fc2_out + NUM_ASSETS, NUM_ASSETS)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.float().to(self.device)

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
        hidden_state = hidden_state.permute(1, 0, 2).squeeze(1)
        action = F.relu(self.fc1(hidden_state))
        action = torch.flatten(action, start_dim=1, end_dim=-1)
        action = F.relu(self.fc2(action))
        action = torch.cat((action, action_w_1), dim=-1)
        action = self.fc3(action).squeeze()
        action = action / torch.norm(action, p=2, dim=-1, keepdim=True)
        action = F.softmax(action, dim=-1)
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
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticLSTM(nn.Module):
    def __init__(self, args, embedding):
        super(CriticLSTM, self).__init__()
        self.args = args
        self.timeenc = "timef"
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
