import torch
import numpy as np

from data_management.environment import Environment
from agent.networks import CriticNetwork, ActorNetwork
from agent.agent import Agent


NUM_FEATURES = 3
NUM_PERIODS = 50
NUM_ASSETS = 8
GRANULARITY = 900
input_dims = [NUM_FEATURES, NUM_PERIODS, NUM_ASSETS]


def main():

    """
    Noise
    """
    # action = np.zeros(NUM_ASSETS)
    # noise = OUActionNoise(action)
    # print(noise())
    """
    ReplayBuffer
    """
    # state1 = np.random.rand(input_dims)
    # state2 = np.random.rand(8)
    #
    # state = (state1, state2)
    # action = np.random.rand(8)
    # reward = 0.00232

    # mem = ReplayBuffer(1000, (input_dims), NUM_ASSETS)
    # mem.store_transition(state, action, reward, state, False)
    # mem.store_transition(state, action, reward, state, False)
    # mem.store_transition(state, action, reward, state, False)

    # a, b, c, d, e = mem.sample_buffer(2)
    """
    Network
    """
    actor = ActorNetwork(alpha=0.01, input_dims=input_dims, name="Actor")
    state = (torch.randn((3, 50, 8)), torch.randn(8))
    action_1 = torch.randn(8)
    action = actor(state)
    print(action)

    critic = CriticNetwork(beta=0.01, input_dims=input_dims, name="Critic")
    q = critic(state, action_1)
    print(q)

    """
    Agent
    """
    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=input_dims,
        tau=0.001,
        # env=env,
        n_actions=NUM_ASSETS,
        batch_size=64,
    )
    start_date = "2022-09-30-00-00"
    env = Environment(
        num_features=NUM_FEATURES,
        num_periods=NUM_PERIODS,
        granularity=GRANULARITY,
        start_date=start_date,
    )

    obs = env.reset()
    action = agent.choose_action(obs)
    print(action)

    return 0


if __name__ == "__main__":
    main()
