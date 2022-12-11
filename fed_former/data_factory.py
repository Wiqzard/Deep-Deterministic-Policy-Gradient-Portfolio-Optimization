import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
from data_management.data_manager import PriceHistory
from agent.time_features import time_features
from sklearn.preprocessing import StandardScaler


from utils.constants import *


class ActionMemory:
    def __init__(self, len) -> None:

        self.action_memory = None
        self.len = len
        self.init_actions()

    def init_actions(self):
        init_action = NUM_ASSETS * [1 / NUM_ASSETS]
        self.action_memory = np.array(self.len * [init_action])

    def store_action(self, actions, idxs):
        for idx, action in zip(idxs, actions):
            self.action_memory[idx] = action

    def get_action(self, idx):
        return self.action_memory[idx]


class DataSet(Dataset):
    def __init__(self, args, flag) -> None:
        super().__init__()
        self.args = args
        self.price_history = PriceHistory(
            args,
            num_periods=args.seq_len,
            granularity=args.granularity,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        self.action_memory = ActionMemory(len(self.price_history))

        self.label_len = 0
        self.pred_len = 0

        self.indices = list(range(len(self.price_history)))
        self.current_index = 0

        self.close_prices = (
            self.price_history.filled_feature_matrices[0].iloc[:, 1:].values
        )

        self.timeenc = 1
        self.__set_data_stamp()

        self.scaler = StandardScaler()
        # self.scaled_close_prices = self.get_scaled_close_prices()

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler_parameters(self):
        return self.scaler.mean_, self.scaler.var_

    def __len__(self) -> None:
        return len(self.close_prices) - self.args.seq_len - self.pred_len - 1

    def get_state(self, index):
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.scaled_close_prices[s_begin:s_end, :]
        # seq_y = self.scaled_close_prices[r_begin:r_end, :]
        seq_x = self.close_prices[s_begin:s_end, :]
        seq_y = self.close_prices[r_begin:r_end, :]
        # scaler = StandardScaler()
        # scaler.fit(seq_x)
        # seq_x = scaler.transform(seq_x)
        # seq_x = self.scaler.fit_transform(seq_x)
        # mu, sigma = scaler.mean_, scaler.var_  # self.get_scaler_parameters()
        #        seq_y = self.scaler.fit_transform(seq_y)
        ## seq_x_mark = self.data_stamp[:, s_begin:s_end]
        # seq_y_mark = self.data_stamp[:, r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        mu = seq_x.mean(0, keepdim=True).float()
        sigma = seq_x.std(0, unbiased=False, keepdim=True).float()
        seq_x -= mu
        seq_x /= sigma

        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return (mu, sigma), (seq_x, seq_y, seq_x_mark, seq_y_mark)

    def __getitem__(self, index: int) -> np.array:
        scales, state = self.get_state(index)
        action_t_1 = torch.tensor(
            self.action_memory.get_action(index), dtype=torch.float32
        )
        next_state = self.get_state(index + 1)
        return index, scales, state, action_t_1, next_state

    def sample_batch(self, batch_size):
        if self.current_index + batch_size > len(self):
            self.current_index = 0
            random.shuffle(self.indices)

        (
            batch_state_x,
            batch_state_mark,
            batch_prev_action,
            batch_next_state_x,
            batch_next_state_mark,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(self.current_index, self.current_index + batch_size):
            state, prev_action, next_state = self[i]
            batch_state_x.append(state[0])
            batch_state_mark.append(state[2])
            batch_prev_action.append(prev_action)
            batch_next_state_x.append(next_state[0])
            batch_next_state_mark.append(next_state[2])

        batch = (
            (torch.stack(batch_state_x), torch.stack(batch_state_mark)),
            torch.stack(batch_prev_action),
            (torch.stack(batch_next_state_x), torch.stack(batch_next_state_mark)),
        )
        self.current_index += batch_size
        return list(range(self.current_index, self.current_index + batch_size)), batch

    def __set_data_stamp(self) -> None:
        df_stamp = self.price_history.filled_feature_matrices[0].iloc[:, [0]]
        df_stamp["date"] = pd.to_datetime(df_stamp["time"])
        data_stamp = time_features(df_stamp[["date"]], timeenc=self.timeenc, freq="t")
        data_stamp = data_stamp  # .transpose(1,0)
        self.data_stamp = data_stamp

    def get_scaled_close_prices(self) -> None:
        self.scaler.fit(self.close_prices)
        closes = self.scaler.transform(self.close_prices)
        return closes
