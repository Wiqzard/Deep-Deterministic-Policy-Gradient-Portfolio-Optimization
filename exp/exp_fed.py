
import numpy as np
from tqdm import tqdm
import torch
from fed_former.agent_ts import Agent
from fed_former.data_factory import DataSet
from fed_former.lstm import ActorLSTM
from torch.utils.data import DataLoader
import torch.optim as optim

#
#    args.batch_size, args_shuffle, args.drop_last 
#    args.d_model, args.embed_type

class Exp_Fed:
    def __init__(self, args) -> None:
        self.args = args
        self.train_data = DataSet(args, flag="train")
        self.test_data = DataSet(args, flag="test") 
    
        self.get_embedding
        self.actor = ActorLSTM(args, embed_type=args.embed_type, freq="t")

    def get_data(flag: str) -> None:
        pass
        
    
    def get_datalaoder(self, flag:str) -> DataLoader:
        args = self.args
        if flag=="train":
            return  DataLoader(self.train_data, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)
        elif flag=="test":
            return  DataLoader(self.test_data, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.drop_last)
    
    def get_optimizer(self):
        # sourcery skip: assign-if-exp, inline-immediately-returned-variable
        if self.args.optim == "adam":
            optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate)
        else:
            optimizer = None
        return  optimizer

    def learn(self):
        dataloader = self.get_dataloader("train")
        optimizer = self.get_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GrandScaler()
            
        self.actor.train()
        for eposide in range(self.args.episodes):
            for idxs, scales, states, prev_actions, _ in tqdm(dataloader, total=len(dataloader), leave=True)
                states, _, state_time_marks, _ = states

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        actions = self.actor(states, state_time_marks, prev_actions) 
                else:
                    actions = self.actor(states, state_time_marks, prev_actions) 
                
                rewards = calculate_rewards_torch(scales, states, prev_actions, actions, self.args)
                reward = calculate_cummulative_reward(rewards) 

                if self.args.use_amp:
                    scaler.scale(reward).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reward.backward()
                    optimizer.step()
                optimizer.zero_grad() 
            
            
            
            
            