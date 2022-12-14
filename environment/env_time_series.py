import numpy as np
import math
from typing import Tuple, List, Dict
from utils.constants import *
from utils.tools import train_test_split

from fed_former.data_factory import DataSet


class Environment:
    def __init__(self, args, flag="train") -> None:
        self.args = args
        self._set_dates(flag)

        self.state_space = DataSet(args, flag)

        self.period = 0
        self.start_action = (
            np.array([1] + (NUM_ASSETS - 1) * [0])
            if args.use_numeraire
            else np.array(NUM_ASSETS * [1 / NUM_ASSETS])
        )
        self.reward_history = []
        self.state_history = []
        self.action_history = []

    def _set_dates(self, flag) -> None:
        (
            start_date_train,
            end_date_train,
            start_date_test,
            end_date_test,
        ) = train_test_split(
            self.args.ratio,
            self.args.granularity,
            self.args.start_date,
            self.args.end_date,
        )
        self.start_date = start_date_train if flag == "train" else start_date_test
        self.end_date = end_date_train if flag == "train" else end_date_test
        if flag == "full":
            self.start_date = self.args.start_date
            self.end_date = self.args.end_date

    def get_numeraire_ratio(self):
        first_value = (
            self.state_space.filled_feature_matrices[0].iloc[:, 1:].values[0, 0]
        )
        last_value = (
            self.state_space.filled_feature_matrices[0].iloc[:, 1:].values[-1, 0]
        )

        return (last_value / first_value).item()

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

        if self.args.use_numeraire:
            y_t = X_t[self.args.seq_len, 1:] / X_t[self.args.seq_len - 1, 1:]
            y_t = np.concatenate((np.ones((1)), y_t))
        else:
            y_t = X_t[self.args.seq_len, :] / X_t[self.args.seq_len - 1, :]

        w_t_prime = (np.multiply(y_t, w_t_1)) / np.dot(y_t, w_t_1)
        mu_t = 1 - self.args.commission_rate_selling * sum(
            np.abs(w_t_prime - state_action["action_t"])
        )
        r_t = math.log(mu_t * np.dot(y_t, w_t_1))
        self.reward_history.append(r_t)
        return r_t

    def reset(self) -> Tuple[Tuple, int]:
        """
        Reset attributes
        retun start state
        """
        self.period = 0
        seq_x, _, seq_x_mark, _ = self.state_space[self.period]
        start_state = seq_x, self.start_action

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
        seq_x, _, seq_x_mark, _ = self.state_space[self.period]
        start_state = seq_x, self.start_action

        curr_states_action = {
            "state_t": (
                seq_x,
                self.action_history[-1],
            ),
            "action_t": action,
        }
        next_state = (seq_x, seq_x_mark, action)
        reward = self.calculate_reward(curr_states_action)
        done = self.period + self.state_space.num_periods == len(self.state_space) - 1
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.state_history.append(next_state)
        return next_state, reward, done
