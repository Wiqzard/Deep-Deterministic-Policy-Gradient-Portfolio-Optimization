import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from agent.noise import OUActionNoise
from agent.replay_buffer import ReplayBuffer
from agent.networks import CriticNetwork, ActorNetwork

from utils.constants import *
from utils.tools import logger


class Agent(object):
    def __init__(self, args, flag="train"):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flag = flag
        self.memory = ReplayBuffer(args)
        # self.memory2 = ReplayBuffer(args, max_size=10000)

        self.critic = CriticNetwork(args)
        self.actor = ActorNetwork(args)
        if args.colab:
            self.target_critic = CriticNetwork(args)
            self.target_actor = ActorNetwork(args)
            self.update_network_parameters(tau=1)
        else:
            self.target_critic = copy.deepcopy(self.critic)
            self.target_actor = copy.deepcopy(self.actor)
        self._create_checkpoint_files()

        if args.noise == "OU":
            self.noise = OUActionNoise(args=args, mu=np.zeros(NUM_ASSETS))
        if args.noise == "param":
            self.actor_noised = ActorNetwork(args)
            self.scalar = 0.05
            self.desired_distance = 0.1
            self.scalar_decay = 0.992

        self.MSE = nn.MSELoss()

    def __add_dim(self, array: np.array) -> torch.Tensor:
        tensor = torch.tensor(array).float().to(self.device)
        return tensor.unsqueeze(0) if len(tensor.shape) in {1, 3} else tensor

    def choose_action(self, oberservation, flag="train"):
        # sourcery skip: extract-method, inline-immediately-returned-variable, merge-duplicate-blocks, remove-redundant-if
        self.actor.eval()
        oberservation = (
            self.__add_dim(oberservation[0]),
            self.__add_dim(oberservation[1]),
        )
        with torch.no_grad():
            mu = self.actor(oberservation).to(self.device)
            # print(f"THE FOLLOWING IS FROM CHOOSE ACTION: {mu}")
            if flag == "train":
                if self.args.noise == "OU":
                    noise = torch.tensor(self.noise()).float().to(self.device)
                    noise = torch.clip(
                        noise, -1.5 * self.args.sigma, 1.5 * self.args.sigma
                    )
                    mu_prime = F.softmax(mu + noise)
                    # mu_prime = F.softmax(mu)
                elif self.args.noise == "param":
                    self.actor_noised.eval()
                    self.actor_noised.load_state_dict(self.actor.state_dict().copy())
                    self.actor_noised.add_parameter_noise(self.scalar)
                    action_noised = self.actor_noised(oberservation).to(self.device)
                    distance = torch.sqrt(torch.mean(torch.square(mu - action_noised)))
                    if distance > self.args.desired_distance:
                        self.scalar *= self.args.scalar_decay
                    if distance < self.args.desired_distance:
                        self.scalar /= self.args.scalar_decay
                    mu_prime = action_noised

                elif self.args.noise == "randn":
                    #    #noise = torch.abs(torch.randn_like(mu)*self.args.sigma).to(self.device)
                    #    noise = (torch.randn_like(mu)*self.args.sigma).to(self.device)
                    #    mu_prime = mu + noise
                    #    #mu_prime /= torch.sum(mu_prime)
                    #    mu_prime = nn.functional.softmax(mu_prime)
                    if random.random() < 0.1:
                        mu_prime = self.random_action().float().to(self.device)
            else:
                mu_prime = F.softmax(mu)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done, flag=None):
        if flag == "bug":
            self.memory2.store_transition(state, action, reward, new_state, done)
        else:
            self.memory.store_transition(state, action, reward, new_state, done)

    def random_action(self):
        action = torch.tensor(NUM_ASSETS * [0])
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        noise = torch.abs(torch.randn_like(action.float()) * self.args.sigma)
        action = action + noise
        action = action / sum(action)
        return action

    def fix_const_outputs(self):
        """A function with no meaning, that fixes the biggest bug in history"""
        # if self.memory2.mem_cntr < 2:  # self.args.batch_size:
        #    return
        # print("------------------------- bullshit ---------------------------")
        # state, _, _, _, _ = self.memory.sample_buffer(2)
        # state = (
        #    torch.tensor(state[0]).float().to(self.critic.device),
        #    torch.tensor(state[1]).float().to(self.critic.device),
        # )
        state = (
            torch.rand((2, NUM_FEATURES, self.args.seq_len, NUM_ASSETS)).to(
                self.device
            ),
            torch.rand((2, NUM_ASSETS)).to(self.device),
        )

        with torch.no_grad():
            mu = self.actor(state)

    def learn(self):
        if self.memory.mem_cntr < self.args.batch_size:
            return
        # print("------------------------- learn ---------------------------")
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.args.batch_size
        )
        reward_multiplier = self.args.reward_multiplier
        reward = torch.tensor(reward).float().to(self.critic.device) * reward_multiplier
        done = torch.tensor(done).float().to(self.critic.device)
        new_state = (
            torch.tensor(new_state[0]).float().to(self.critic.device),
            torch.tensor(new_state[1]).float().to(self.critic.device),
        )
        action = torch.tensor(action).float().to(self.critic.device)
        state = (
            torch.tensor(state[0]).float().to(self.critic.device),
            torch.tensor(state[1]).float().to(self.critic.device),
        )
        # self.target_actor.eval()
        # self.target_critic.eval()
        # self.critic.eval()
        if self.args.use_amp:
            critic_scaler = torch.cuda.amp.GradScaler()
            actor_scaler = torch.cuda.amp.GradScaler()

        # <---------------------------- update critic ----------------------------> #

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                target_actions = self.target_actor(new_state)
                critic_value_ = self.target_critic(new_state, target_actions)
                critic_value = self.critic(state, action)
        else:
            target_actions = self.target_actor(new_state)
            critic_value_ = self.target_critic(new_state, target_actions)
            critic_value = self.critic(state, action)

        target = [
            reward[j] + self.args.gamma * critic_value_[j] * (1 - done[j])
            for j in range(self.args.batch_size)
        ]
        target = torch.tensor(target).float().to(self.critic.device)
        target = target.view(self.args.batch_size, 1).squeeze()

        critic_loss = self.MSE(target, critic_value)

        self.critic.zero_grad()

        if self.args.use_amp:
            critic_scaler.scale(critic_loss).backward()
            critic_scaler.step(self.critic.optimizer)
            critic_scaler.update()
        else:
            critic_loss.backward()
            self.critic.optimizer.step()

        # <---------------------------- update actor ----------------------------> #

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                mu = self.actor(state)
        else:
            mu = self.actor(state)

        actor_loss = -self.critic(state, mu)
        actor_loss = torch.mean(actor_loss)

        self.actor.zero_grad()
        if self.args.use_amp:
            actor_scaler.scaler(actor_loss).backward()
            actor_scaler.step(self.actor.optimizer)
            actor_scaler.update()
        else:
            actor_loss.backward()
            self.actor.optimizer.step()

        self.update_network_parameters()

        with open("log_loss.txt", "a+") as f:
            f.write(f"Actor loss: {actor_loss.item()}\n")
            f.write(f"Critic loss: {critic_loss.item()}\n")

    def update_network_parameters(self, tau=None):
        self.tau = self.args.tau
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

    def save_models(self) -> None:
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self) -> None:
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def _create_checkpoint_files(self) -> None:
        self.actor.create_checkpoint(name="actor")
        self.target_actor.create_checkpoint(name="target_actor")
        self.critic.create_checkpoint(name="critic")
        self.target_critic.create_checkpoint(name="target_critic")
