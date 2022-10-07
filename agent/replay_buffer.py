import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = (
            np.zeros((self.mem_size, *input_shape)),
            np.zeros((self.mem_size, n_actions)),
        )  # * unpacks tuple
        self.new_state_memory = (
            np.zeros((self.mem_size, *input_shape)),
            np.zeros((self.mem_size, n_actions)),
        )
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[0][index] = state[0]
        self.state_memory[1][index] = state[1]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[0][index] = state_[0]
        self.new_state_memory[1][index] = state_[1]
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(
            max_mem, batch_size
        )  # Generate a uniform random sample from np.arange(max_mem) of size batch_size:

        states = (self.state_memory[0][batch], self.state_memory[1][batch])
        new_states = (self.new_state_memory[0][batch], self.new_state_memory[1][batch])
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


state1 = np.random.rand(3, 50, 8)
state2 = np.random.rand(8)

state = (state1, state2)
action = np.random.rand(8)
reward = 0.00232

mem = ReplayBuffer(1000, (3, 50, 8), 8)
mem.store_transition(state, action, reward, state, False)
mem.store_transition(state, action, reward, state, False)
mem.store_transition(state, action, reward, state, False)

a, b, c, d, e = mem.sample_buffer(2)
print(a[0].shape)
print(a[1].shape)
print(b.shape)
print(c.shape)
print(d[0].shape)
print(d[1].shape)
print(e.shape)
