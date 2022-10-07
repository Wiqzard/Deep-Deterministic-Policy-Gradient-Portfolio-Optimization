from data_management.data_manager import *

import math


class Environment:
    def __init__(
        self,
        num_features: int,
        num_periods: int,
        granularity: int,
        start_date: str,
        end_date: str = None,
    ):

        self.num_features = num_features
        self.num_periods = num_periods

        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.commision_rate_selling = 0.0025
        self.commision_rate_purchasing = 0.0025
        self.state_space = PriceHistory(
            num_features=self.num_features,
            num_periods=self.num_periods,
            granularity=self.granularity,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.state_space.set_data_matrix()

        self.period = 1

        self.reward_history = []
        self.state_history = []

        self.state_buffer = None
        self.action_buffer = None

    def calculate_reward(self, state_action: dict) -> float:
        """
        args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
        -> Calculate y_t from X_t = v_t // v_t-1
        -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
        -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
        -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
        """
        # [feature_number, num_periods before t ascending, num_assets] -> -2:
        X_t = state_action["state_t"][0]
        w_t_1 = state_action["state_t"][1]
        y_t = np.reciprocal(
            X_t[0, -2, :]
        )  # np.ones_like(state_action["state_t"][0, -2:, :]) / state_action["state_t"][0, -2:, :]
        w_t_prime = (np.multiply(y_t, w_t_1)) / np.dot(y_t, w_t_1)
        mu_t = self.commision_rate_selling * sum(
            np.abs(w_t_prime - state_action["action_t"])
        )
        r_t = 1 / self.period * math.log(mu_t * np.dot(y_t, w_t_1))
        # rate_of_return = mu_t* np.dot(y_t, w_t_1)-1
        self.reward_history.append(r_t)
        return r_t

    def _calculate_cummulated_reward(self, states_actions: dict) -> float:
        assert states_actions

        # action = list [num_assets]
        # y_t = states_actions[]
        # total_reward = 1/len(states_actions)
        # return total_reward

    def reset(self):
        """
        Reset attributes
        retun start state
        """
        self.reward_history = []
        self.state_history = []
        self.period = 1
        start_action = np.zeros(self.state_space.num_assets)
        start_action[0] = 1
        start_state = (
            self.state_space.normalized_price_matrix(self.period),
            np.empty(self.state_space.num_assets),
        )
        self.state_buffer = start_state
        self.action_buffer = start_action

        return start_state

    def step(self, action):
        """
        args: action_t
        output: X_t+1, reward_t, done, info
        """
        self.period += 1
        self.action_buffer = action
        curr_states_action = {
            "state_t": (
                self.state_space.normalized_price_matrix(self.period),
                self.action_buffer,
            ),
            "action_t": action,
        }  # For the rewards
        next_state = (self.state_space.normalized_price_matrix(self.period), action)
        self.state_buffer = next_state
        reward = self.calculate_reward(curr_states_action)  # r_t(s_t=(X_t, a_t-1), a_t)
        done = (
            self.period + self.state_space.num_periods
            == self.state_space.data_matrix[0].shape[0]
        )  # True if end of sequence

        return next_state, reward, done


granularity = 900
start_date = "2022-09-30-00-00"
env = Environment(num_features=3, num_periods=7, granularity=900, start_date=start_date)

done = False
total_reward = 0
state = env.reset()
while not done:
    rand = np.random.rand(2)
    action = rand / sum(rand)
    state, reward, done = env.step(action)
    # total_reward += reward

print(state)
print(reward)
print(done)
print(env.period)
