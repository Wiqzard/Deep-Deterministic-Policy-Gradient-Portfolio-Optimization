import torch
import numpy as np

from data_management.coin_database import CoinDatabase
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
    Database
    """
    # base = CoinDatabase()
    # base.create_all_tables()
    # base.fill_all_tables(granularity=900, start_date="2022-10-5-00-00", end_date=None)

    #    test_shape_base = base.get_coin_data(
    #       coin="BTC-USD", granularity=900, start_date="2022-10-5-00-00", end_date=None
    #  ).shape[0]
    # print(test_shape_base)

    """
    ENV
    """
    #    done = False
    #    total_reward = 0
    #    state = env.reset()
    #    total_reward = 0
    #    while not done:
    #        rand = np.random.rand(2)
    #        action = rand / sum(rand)
    #        state, reward, done = env.step(action)
    #        total_reward += reward
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
    # actor = ActorNetwork(alpha=0.01, input_dims=input_dims, name="Actor")
    # state = (torch.randn((3, 50, 8)), torch.randn(8))
    # action_1 = torch.randn(8)
    # action = actor(state)
    # print(action)

    # critic = CriticNetwork(beta=0.01, input_dims=input_dims, name="Critic")
    # q = critic(state, action_1)
    # print(q)
    # state = (torch.rand((64, 3, 50, 2)), torch.rand((64, 1, 1, 2)))
    # action = torch.rand(64, 1, 1, 2)
    # critic = CriticNetwork(beta=0.01, input_dims=[3, 50, 2], name="Critic")
    # critic.eval()
    # state_value = critic(state, action)
    # print(state_value.shape)
    """
    Agent
    """
    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=input_dims,
        tau=0.001,
        # env=env,
        n_actions=NUM_ASSETS,  # NUM_ASSETS,
        batch_size=64,
    )
    start_date = "2022-08-30-00-00"

    data_base = CoinDatabase()
    data_base.create_all_tables()
    data_base.fill_all_tables(
        granularity=900, start_date="2022-10-5-00-00", end_date=None
    )

    env = Environment(
        num_features=NUM_FEATURES,
        num_periods=NUM_PERIODS,
        granularity=GRANULARITY,
        start_date=start_date,
        data_base=data_base,
    )

    score_history = []
    np.random.seed(0)
    for _ in range(50):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        print(score)
        score_history.append(score)

    return 0


if __name__ == "__main__":
    main()
