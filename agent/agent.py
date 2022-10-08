import torch
import torch.nn.functional as F
import numpy as np
from agent.noise import OUActionNoise
from agent.replay_buffer import ReplayBuffer
from agent.networks import CriticNetwork, ActorNetwork


class Agent(object):
    def __init__(
        self,
        alpha: float,
        beta: float,
        input_dims: list[int, int, int],
        tau: float,
        # env,
        n_actions: int,
        gamma=0.99,
        max_size=1000000,
        batch_size=64,
    ):
        """
        critic
        beta, num_features, num_periods, num_assets,name

        replay
        self, max_size, input_shape, n_actions
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # input
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, name="Actor")
        self.target_actor = ActorNetwork(alpha, input_dims, name="TargetActor")
        self.critic = CriticNetwork(beta, input_dims, name="Critic")
        self.target_critic = CriticNetwork(beta, input_dims, name="TargetCritic")

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, oberservation):
        self.actor.eval()
        # oberservation = torch.tensor(oberservation).to(self.actor.device)
        oberservation = (
            torch.tensor(oberservation[0], dtype=torch.float),
            torch.tensor(oberservation[1], dtype=torch.float).unsqueeze(0).unsqueeze(0),
        )
        mu = self.actor(oberservation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(
            self.actor.device
        )
        mu_prime = mu_prime / torch.sum(mu_prime)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)
        new_state = (
            torch.tensor(new_state[0], dtype=torch.float).to(self.critic.device),
            torch.tensor(new_state[1], dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.critic.device),
        )
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state = (
            torch.tensor(state[0], dtype=torch.float).to(self.critic.device),
            torch.tensor(state[1], dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.critic.device),
        )

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = [
            reward[j] + self.gamma * critic_value_[j] * done[j]
            for j in range(self.batch_size)
        ]

        target = torch.tensor(target, dtype=torch.float).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

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
