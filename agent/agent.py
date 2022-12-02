import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from agent.noise import OUActionNoise
from agent.replay_buffer import ReplayBuffer
from agent.networks import CriticNetwork, ActorNetwork


class Agent(object):
    def __init__(self,config, flag="train"):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flag = flag
        self.memory = ReplayBuffer(config)

        self.actor = ActorNetwork(config)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = CriticNetwork(config)
        self.target_critic = copy.deepcopy(self.critic)

        self.noise = OUActionNoise(config=config, mu=np.zeros(config.num_assets))

        #self.update_network_parameters(tau=self.tau)
        self.MSE = nn.MSELoss()
    
    def __add_dim(self, array:np.array) -> torch.Tensor:
        tensor = torch.tensor(array).float().to(self.device)
        return tensor.unsqueeze(0) if len(tensor.shape) in {1, 3} else tensor

    def choose_action(self, oberservation, flag="train"):
        self.actor.eval()
        oberservation = (
            self.__add_dim(oberservation[0]),
            self.__add_dim(oberservation[1])
            )
        mu = self.actor(oberservation).to(self.device)
        if self.flag == "train" and self.config.noise == "OU":
            noise = torch.tensor(self.noise()).float().to(self.device)
            mu_prime = mu + noise
            mu_prime = nn.functional.softmax(mu_prime)
        elif self.flag == "train" and self.config.noise == "randn": 
            #noise = torch.abs(torch.randn_like(mu)*self.config.sigma).to(self.device) 
            noise = (torch.randn_like(mu)*self.config.sigma).to(self.device) 
            mu_prime = mu + noise
            #mu_prime /= torch.sum(mu_prime)
            mu_prime = nn.functional.softmax(mu_prime)
        else:
            mu_prime = mu
        self.actor.train()
        return mu_prime.cpu().detach().numpy()########

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.config.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.config.batch_size
        )
        
        reward = torch.tensor(reward).float().to(self.critic.device)
        done = torch.tensor(done).float().to(self.critic.device)
        new_state = (
            torch.tensor(new_state[0]).float().to(self.critic.device),
            torch.tensor(new_state[1]).float()
            .to(self.critic.device),
        )
        action = (
            torch.tensor(action).float()
            .to(self.critic.device)
        )
        state = (
            torch.tensor(state[0]).float().to(self.critic.device),
            torch.tensor(state[1]).float()
            .to(self.critic.device),
        )
        #self.target_actor.eval()
        #self.target_critic.eval()
        #self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = [
            reward[j] + self.config.gamma * critic_value_[j] * (1-done[j])
            for j in range(self.config.batch_size)
        ]
        target = torch.tensor(target).float().to(self.critic.device)
        target = target.view(self.config.batch_size, 1).squeeze()

        #self.critic.train()i
        self.critic.zero_grad()#.optimizer.zero_grad()

        critic_loss = self.MSE(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        #self.critic.eval()
        self.actor.zero_grad()#optimizer.zero_grad()
        mu = self.actor.forward(state)#.unsqueeze(1).unsqueeze(1)
        #self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        self.tau = self.config.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
