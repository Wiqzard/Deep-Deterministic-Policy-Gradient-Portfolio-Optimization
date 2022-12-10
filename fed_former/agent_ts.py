import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from agent.noise import OUActionNoise
from utils.constants import *
from utils.tools import logger

from fed_former.lstm import ActorLSTM, CriticLSTM
from fed_former.layers.embeddings import DataEmbedding


class Agent(object):
    def __init__(self, args, flag="train"):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flag = flag

        self.embedding = DataEmbedding(
            c_in=NUM_ASSETS, d_model=args.d_model, embed_type="timef", freq="t"
        )
        self.embedding_target = DataEmbedding(
            c_in=NUM_ASSETS, d_model=args.d_model, embed_type="timef", freq="t"
        )
        self.critic = CriticLSTM(args, self.embedding)
        self.actor = ActorLSTM(args, self.embedding)
        self.target_critic = CriticLSTM(args, self.embedding_target)
        self.target_actor = ActorLSTM(args, self.embedding_target)
        self.update_network_parameters(tau=1)

        self._create_checkpoint_files()

        if args.noise == "OU":
            self.noise = OUActionNoise(args=args, mu=np.zeros(NUM_ASSETS))
        self.MSE = nn.MSELoss()

    def _create_checkpoint_files(self) -> None:
        self.actor.create_checkpoint(name="actor")
        self.target_actor.create_checkpoint(name="target_actor")
        self.critic.create_checkpoint(name="critic")
        self.target_critic.create_checkpoint(name="target_critic")

    def choose_action(self, states, prev_actions, flag="train"):
        self.actor.eval()

        seq_x, seq_x_mark = states
        action = prev_actions

        with torch.no_grad():
            mu = self.actor(
                state=seq_x, time_mark=seq_x_mark, prev_action=action, hidden=None
            ).to(self.device)
            if flag == "train":
                if self.args.noise == "OU":
                    noise = torch.tensor(self.noise()).float().to(self.device)
                    noise = torch.clip(noise, -self.args.sigma, self.args.sigma)
                    # mu_prime = F.softmax(mu + noise)
                    mu_prime = mu
                    # print(noise)
                    # mu_prime = torch.abs(mu + noise)
                    # print(mu_prime)
                    # mu_prime = mu_prime / sum(mu_prime)

                elif self.args.noise == "randn":
                    if random.random() < 0.1:
                        mu_prime = self.random_action().float().to(self.device)
                else:
                    mu_prime = mu
            else:
                mu_prime = mu
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def random_action(self):
        action = torch.tensor(NUM_ASSETS * [0])
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        action[random.randint(0, NUM_ASSETS - 1)] = 1
        noise = torch.abs(torch.randn_like(action.float()) * self.args.sigma)
        action = action + noise
        action = action / sum(action)
        return action

    def learn(self, states, prev_actions, actions, next_states, rewards):
        """watch out with torch tensor tuples"""

        states_x, states_time_mark = states
        next_states_x, next_states_time_mark = next_states
        # self.target_actor.eval()
        # self.target_critic.eval()
        # self.critic.eval()
        if self.args.use_amp:
            critic_scaler = torch.cuda.amp.GradScaler()
            actor_scaler = torch.cuda.amp.GradScaler()

        # <---------------------------- update critic ----------------------------> #
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                target_actions = self.target_actor.forward(*next_states, actions)
                critic_value_ = self.target_critic.forward(
                    *next_states, actions, target_actions
                )
                critic_value = self.critic.forward(states, prev_actions, actions)
        else:
            target_actions = self.target_actor.forward(
                next_states_x, next_states_time_mark, prev_action=actions
            )
            critic_value_ = self.target_critic.forward(
                next_states_x,
                next_states_time_mark,
                prev_action=actions,
                action=target_actions,
            )
            critic_value = self.critic.forward(
                states_x, states_time_mark, prev_actions, action=actions
            )
        target = [
            rewards[j] + self.args.gamma * critic_value_[j]
            for j in range(self.args.batch_size)
        ]
        target = torch.tensor(target).float().to(self.device)
        target = target.view(self.args.batch_size, 1).squeeze()

        critic_loss = self.MSE(target, critic_value)

        self.critic.zero_grad()
        if self.args.use_amp:
            critic_scaler.scale(critic_loss.float()).backward()
            critic_scaler.step(self.critic.optimizer)
            critic_scaler.update()
        else:
            critic_loss.backward()
            self.critic.optimizer.step()

        # <---------------------------- update actor ----------------------------> #

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                mu = self.actor.forward(*states, prev_actions)
        else:
            mu = self.actor.forward(*states, prev_actions)

        actor_loss = -self.critic.forward(*states, prev_actions, mu)
        actor_loss = torch.mean(actor_loss)
        # print(actor_loss)

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
