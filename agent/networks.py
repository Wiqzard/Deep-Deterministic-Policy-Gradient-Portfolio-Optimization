import torch
import torch.nn as nn
import torch.optim as optim
import os

from utils.constants import * 


class CriticNetwork(nn.Module):
    def __init__(self, args, name):
        super(CriticNetwork, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=20, kernel_size=(args.seq_len - 2, 1)
        )
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))

        self.bn1 = nn.LayerNorm([1, 1, NUM_ASSETS])

        self.linear_state1 = nn.Linear(NUM_ASSETS, 16)
        #self.linear_state2 = nn.Linear(16, 16)

        self.linear_action = nn.Linear(NUM_ASSETS, 16)
        self.bn2 = nn.LayerNorm(16)
        self.linear_q = nn.Linear(16, 1)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        chkpt_dir = args.chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg") 

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)

    def forward(self, state, action):
        action_1 = state[1].unsqueeze(-2).to(self.device)  #torch.Size([10, 1, 8])
        action = action.to(self.device)
        state_value = state[0].to(self.device) #torch.Size([10, 3, 50, 8])
        state_value = self.relu(self.conv1(state_value)) #torch.Size([10, 2, 48, 8])
        state_value = self.relu(self.conv2(state_value)).squeeze(-2)  #torch.Size([10, 20,  8])
        state_value = torch.cat((state_value, action_1), dim=1).unsqueeze(-1) #torch.Size([10, 21, 8, 1])
        state_value = self.conv3(state_value).squeeze() #torch.Size([10, 8])
        state_value = self.linear_state1(state_value)
        #state_value = self.linear_state2(state_value)
        action_value = self.linear_action(action)
        state_action_value = self.relu(torch.add(state_value, action_value))
        state_action_value = self.linear_q(state_action_value)
        return state_action_value.squeeze()


# CNN
class ActorNetwork(nn.Module):
    def __init__(self, args, name):

        super(ActorNetwork, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(
            in_channels=NUM_FEATURES, out_channels=2, kernel_size=(3, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=20, kernel_size=(args.seq_len - 2, 1))

        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))
        self.linear = nn.Linear(in_features=NUM_ASSETS, out_features=NUM_FEATURES)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        chkpt_dir = args.chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg") 

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        torch.load_state_dict(torch.load.checkpoint_file)

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
        #action = self.relu(self.linear(action))
        cash_bias = torch.cat(
            (torch.ones(*action.shape[:-1], 1), 
             torch.zeros(*action.shape[:-1], action.shape[-1]-1)), dim=-1).to(self.device)
        action = torch.add(action, cash_bias)
        action = self.softmax(action)
        return action
