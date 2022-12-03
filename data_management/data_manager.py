import warnings
import pandas as pd
import numpy as np
from numpy import ndarray as ndarray
import datetime
import requests
import json
import time
import datetime
from datetime import datetime, timedelta
from random import randint
from tqdm.auto import tqdm
import logging
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

from data_management.coin_database import CoinDatabase
from utils.constants import *
from agent.time_features import time_features
from utils.tools import count_granularity_intervals, logger


warnings.filterwarnings('ignore')

class PriceHistory(Dataset):
    """
    * price tensor X_t = [feature_number, number_periods, number_assets]
                    = cat((V_t, V^high_t, V^lo_t), 1) 
        With V^x_t = [v_t-n+1 / v_t | ] 
    * states: s_t=(price tensor X_t, action_t-1)
    * action: number_assets x 1 array
    * (state_t, action_t) -> (state_t+1=(price_t+1, action_t), action_t+1)
        -> (state_t+2=(price_t+2, action_t+1), action_t+2) -> ...
    * prices are independent of a
    """ 
    def __init__(
        self,
        args,
        num_periods: int,
        granularity: int = None,
        start_date: str = None,
        end_date: str = None,
        label_len: int=25,
        pred_len: int=25,
        timeenc: int=1,
        scale: bool=False
    ):
        self.coins = COINS
        self.num_assets = NUM_ASSETS
        self.num_periods = num_periods
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.data_matrix = [] 
        self.data_base = CoinDatabase(args)
        self.timeenc = timeenc

        self.label_len = label_len
        self.pred_len = pred_len

        self._check_dates()
        self.__set_data_matrix()
        self.filled_feature_matrices = self.__fill_nan(self.__efficent_all(cash_bias=False))

        self.__set_data_stamp()
        if scale:
          self.filled_feature_matrices_scaled = None
          self.__scale_feature_matrix(feature=0)

           
    def _check_dates(self) -> None:
        data_points = count_granularity_intervals(self.start_date, self.end_date, self.granularity)
        assert self.num_periods <= data_points, f"Not enough time periods in dataset {data_points}, but need {self.num_periods}" 
 

    def __scale_feature_matrix(self, feature:int=0) -> None:
      self.scaler = StandardScaler()
      feature_matrix = self.filled_feature_matrices[feature]
      self.scaler.fit(feature_matrix.iloc[:, 1:].values)
      closes_ = self.scaler.transform(feature_matrix.iloc[:, 1:].values)
      self.filled_feature_matrices_scaled[feature].iloc[:, 1:] = closes_


    # Implement error handling
    def __set_data_matrix(
        self, granularity: int = None, start_date: str = None, end_date: str = None
    ) -> None:
        gran = self.granularity if granularity is None else granularity
        s_date = self.start_date if start_date is None else start_date
        e_date = self.end_date if end_date is None else end_date

        for (
            coin
        ) in self.coins:  # for i, coin in enumerate(self.coins): self.data_matrix[i] =
            data = (
                self.data_base.get_coin_data(
                    coin=coin, granularity=gran, start_date=s_date, end_date=e_date
                )
                if self.data_base
                else self.retrieve_data(
                    ticker=coin, granularity=gran, start_data=s_date, end_date=e_date
                )
            )
            
            self.data_matrix.append(data)


    def get_list_of_npm(self) -> List[pd.DataFrame]:
        return [self.normalized_price_matrix(idx=i) for i in range(1, self.data_matrix[0].shape[0] - self.num_periods)]


    def __efficent_all(self, cash_bias:bool=True) -> List[pd.DataFrame]:  #returns List[normalized_price(close),] for all assets
        closes, highs, lows = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        dates = self.data_matrix[0]["time"]
        closes["time"], highs["time"], lows["time"] = dates, dates, dates

        for i, coin_data in enumerate(self.data_matrix[cash_bias:]):
            closes[f"close_{i}"] = coin_data["close"]
            highs[f"high_{i}"] = coin_data["high"]
            lows[f"low_{i}"] = coin_data["low"]
        return closes, highs, lows


    def __get_nan_sequence_from_column(self, nan_indices) -> Tuple[int, int]:
      """
      input: index column of dataframe, where column.isnan()
      output: index before first NaN, index after last Nan
      """
      index = nan_indices[0] 
      first_value = index - 1 if index > 0 else 0
      temp = first_value
      last_value = index
      for idx in range(len(nan_indices)): #idx in df:
        if nan_indices[idx] == index:
          last_value = index 
          index += 1
        else:
          last_value = index 
          break
      return first_value, last_value


    def __fill_sequence_in_column(self, df: pd.DataFrame, column_name:str):
        """
      input: dataframe and column_name to fill
      output: filled dataframe column
      """
        filled_frame = df.copy()
        nan_indices = df.loc[df[column_name].isna()].index
        first_value_idx, last_value_idx = self.__get_nan_sequence_from_column(nan_indices)
        first_value = df[column_name][first_value_idx]
        last_value = df[column_name][last_value_idx]
        if last_value:
          last_value = first_value
        delta_idx = last_value_idx - first_value_idx
        delta_step = (last_value - first_value) / (delta_idx)
        for i in range(first_value_idx +1 , last_value_idx+1):
            value = first_value + i * delta_step
            value = max(value, 0)
            filled_frame.at[i, column_name] = value
        return filled_frame


    def __one_fill(self, data:pd.DataFrame) -> pd.DataFrame:
      for column in data:
        nan_indices = data.loc[data[column].isna()].index
        if nan_indices.shape[0] != 0:
          filled_sequence = self.__fill_sequence_in_column(data, column)
      return filled_sequence


    def __fill_nan(self, data: Tuple[pd.DataFrame]) -> Tuple[pd.DataFrame]:
        """
      Fills closes, highs, lows completely
      Input: [closes, highs, lows]  (DataFrames) (efficient_all)
      ----------------------------------------------------------------
      take first appaering NaN value and the following last NaN value
      calculate price and index difference
      fill values with last price + delta_price/delta_index + Noise
      """
        closes = data[0]
        highs = data[1]
        lows = data[2]
        first_nan, last_nan = 0, 0
        while closes.isnull().values.any() or highs.isnull().values.any() or lows.isnull().values.any():
            closes = self.__one_fill(closes)
            highs = self.__one_fill(highs)
            lows = self.__one_fill(lows)
        return closes, highs, lows


    def __normalized_feature_matrix1(self, idx:int, feature_matrix: pd.DataFrame) -> np.array:
        """
      Input: index, feature matrix such as closes (with all columns(time included)),
      Output: normalized feature matrix as np array   [1, num_periods, num_cash_free_assets]
      ----------------
      Misc: row of V_t is v_t
      """
        feature_matrix = feature_matrix.iloc[:, 1:]
        V_t = feature_matrix.iloc[idx-50 +1 : idx+1] / feature_matrix.iloc[idx]
        return V_t.to_numpy()[np.newaxis, ...]
    

    def __normalized_feature_matrices(self, idx:int) -> np.array:
        """
      Input: index, feature matrix such as closes (with all columns(time included)),
      Output: normalized feature matrix as np array   [1, num_periods, num_cash_free_assets]
      ----------------
      Misc: row of V_t is v_t
      """
        v_t = self.filled_feature_matrices[0].iloc[idx - 1, 1:].to_numpy()
        return [(feature_matrix.iloc[idx - self.num_periods : idx, 1:] / v_t).to_numpy()[np.newaxis, ...] for feature_matrix in self.filled_feature_matrices]


    def normalized_price_matrix(self, idx:int) -> np.array:
      """
      Input: index, t in v_t
      Output: price_matrix
      """
      cash_bias = 0 # self.filled_feature_matrices[0].shape[1]==self.num_assets-1
      idx += self.num_periods

      assert idx >= self.num_periods 
      assert idx <= self.data_matrix[0].shape[0], f"total length {self.data_matrix[0].shape[0]} but idx={idx}"

      normalized_features_matrices= self.__normalized_feature_matrices(idx=idx)
      num_assets = self.num_assets - cash_bias
      X_t = np.empty((1, self.num_periods, num_assets))
      
      for feature_matrix in normalized_features_matrices:
        X_t = np.concatenate((X_t, feature_matrix), 0)
      X_t = X_t[1:, :, :] #if cash_bias else X_t[1:, :, 1:]
      return X_t.astype(float)


    def __len__(self) -> int:
      return len(self.filled_feature_matrices[0]) - self.num_periods - self.pred_len + 1


    def __set_data_stamp(self):
        # sourcery skip: extract-method, inline-immediately-returned-variable
      df_stamp = self.filled_feature_matrices[0].iloc[:, [0]]
      df_stamp['date'] = pd.to_datetime(df_stamp["time"])
      if self.timeenc == 0:
          df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
          df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
          df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
          df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
          df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
          df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
          data_stamp = df_stamp.drop(['date'], 1).values
      elif self.timeenc == 1:
          data_stamp = time_features(df_stamp[['date']], freq="t")
          data_stamp = data_stamp#.transpose(1,0)
  
      self.data_stamp = data_stamp


    def __getitem__(self, index: int) -> np.array:
      s_begin = index
      s_end = s_begin + self.num_periods
      r_begin = s_end - self.label_len
      r_end = r_begin + self.label_len + self.pred_len
      r_end = r_begin + self.label_len + self.pred_len
      
      seq_x = self.filled_feature_matrices[0].iloc[s_begin:s_end, 1:].values
      seq_y = self.filled_feature_matrices[0].iloc[r_begin:r_end, 1:].values
      #seq_x_mark = self.data_stamp[:, s_begin:s_end]
      #seq_y_mark = self.data_stamp[:, r_begin:r_end]
      seq_x_mark = self.data_stamp[s_begin:s_end, :]
      seq_y_mark = self.data_stamp[r_begin:r_end, :]

      return seq_x, seq_y, seq_x_mark, seq_y_mark


    def inverse_transform(self, data) -> np.array:
      return self.scaler.inverse_transform(data)
    

    def get_return(self, idx:int) -> np.array:
      assert idx < len(self.filled_feature_matrices[0])
      """
      returns relative price vector of closes: v_t / v_t-1, starting at v1 / v0
      """ 
      v_t = self.filled_feature_matrices[0].iloc[idx + idx, 1:].values
      v_t_1 = self.filled_feature_matrices[0].iloc[idx, 1:].values
      return v_t / v_t_1
