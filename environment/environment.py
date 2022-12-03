import numpy as np
import math

from utils.constants import *
from utils.tools import train_test_split
from data_management.data_manager import PriceHistory

class Environment:
    def __init__(self, args, flag="train") -> None:
        self.args = args 
        self._set_dates(flag)
        self.scale_state = False # For now

        self.state_space = PriceHistory(
            args=args,
            num_periods=args.seq_len,
            granularity=args.granularity,
            start_date=self.start_date,
            end_date=self.end_date,
            scale=self.scale_state
        )

        if args.compute_before:
          self.all_npms = self.state_space.filled_feature_matrices
        
        self.period = 0
        self.start_action = [1] + (NUM_ASSETS - 1) * [0]
        self.reward_history = []
        self.state_history = []
        self.action_history = []


    def _set_dates(self, flag) -> None:
        start_date_train, end_date_train, start_date_test, end_date_test = train_test_split(self.args.ratio, self.args.granularity, self.args.start_date, self.args.end_date)
        self.start_date = start_date_train if flag=="train" else start_date_test
        self.end_date = end_date_train if flag=="train" else end_date_test


    @property
    def num_steps(self) -> int:
        return len(self.state_space)


    def calculate_reward(self, state_action: dict) -> float:
        """
        args: dict: {"state_t": (X_t, w_t-1), "action_t" : w_t}
        -> Calculate y_t from X_t = v_t // v_t-1
        -> Calculate w_t' = (y_t*w_t-1) / (y_t.w_t-1)
        -> Calculate mu_t: Assume cp=cs i.e. commission rate for selling and purchasing -> mu_t = c*sum(|\omgega_t,i' - \omega_t,i|)
        -> Calculate r_t = 1/t_f ln(mu_t*y_t . w_t_1)
        """
        X_t = state_action["state_t"][0]
        w_t_1 = state_action["state_t"][1]
        
        y_t = np.reciprocal(    #Correct
            X_t[0, -2, 1:]
        ) 
        y_t = np.concatenate((np.ones((1)), y_t))
        w_t_prime = (np.multiply(y_t, w_t_1)) / np.dot(y_t, w_t_1)
        mu_t = 1 - self.args.commission_rate_selling * sum(
            np.abs(w_t_prime - state_action["action_t"])
        )
        r_t = math.log(mu_t * np.dot(y_t, w_t_1))
        self.reward_history.append(r_t)
        return r_t


    def reset(self):
        """
        Reset attributes
        retun start state
        """
        self.period = 0
        npm = self.state_space.normalized_price_matrix(self.period)
        #npm = self.all_npms[self.period-1] if self.compute_before else self.state_space.normalized_price_matrix(self.period)# cash_bias=True)
        start_state = npm, np.array([1] + [0 for _ in range(self.state_space.num_assets - 1)])
        total_steps = self.num_steps - self.period
        self.reward_history = []
        self.state_history = [start_state]
        self.action_history = [self.start_action]
        return (start_state), total_steps


    def step(self, action):
        """
        args: action_t
        output: X_t+1, reward_t, done, info
        """
        self.period += 1
        npm = self.state_space.normalized_price_matrix(self.period)
        #npm = self.all_npms[self.period-1] if self.compute_before else self.state_space.normalized_price_matrix(self.period)#, cash_bias=True)
        curr_states_action = {
            "state_t": (
                npm,
                self.action_history[-1],
            ),
            "action_t": action,
        }
        next_state = (npm, action)
        reward = self.calculate_reward(curr_states_action)
        done = (
            self.period + self.state_space.num_periods + 1
            == len(self.state_space)
        )
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.state_history.append(next_state)
        return next_state, reward, done


    def calculate_relative_price_episode(self) -> np.array:
        first_closes =  self.state_space[0][0]   
        last_closes = self.state_space[- (self.state_space.num_periods + self.state_space.pred_len+ 5)][1]
        if self.scale_state:
          first_closes = self.state_space.inverse_transform(first_closes)[0,:]
          last_closes = self.state_space.inverse_transform(last_closes)[-1,:]
        else:
          first_closes = first_closes[0, :]
          last_closes = last_closes[-1, :]
        return last_closes / first_closes