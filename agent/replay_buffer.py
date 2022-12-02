import numpy as np
class ReplayBuffer(object):
    """
    n_actions = num_assets
    input state: tuple[np.ndarray(3, 50, 8), np.ndarray(8)]) = tuple[price_matrix, previous_action]
                    -> saves into 2d array (not efficient)

    """
    def __init__(self, config):
        self.config = config
        self.mem_size = config.max_size
        self.n_actions = config.num_assets
        self.num_features = config.num_features
        self.seq_len = config.seq_len
        
        self.mem_cntr = 0
        self.state_memory = (
            np.zeros((self.mem_size, self.num_features, self.seq_len, self.n_actions)),
            np.zeros((self.mem_size, self.n_actions)),
        )  # * unpacks tuple
        self.new_state_memory = (
            np.zeros((self.mem_size, self.num_features, self.seq_len, self.n_actions)),
            np.zeros((self.mem_size, self.n_actions)),
        )
        self.action_memory = np.zeros((self.mem_size, self.n_actions))
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