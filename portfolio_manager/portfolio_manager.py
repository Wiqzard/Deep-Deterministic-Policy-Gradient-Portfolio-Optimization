#@title Portfolio Manager
from typing import Optional
import pandas as pd
import numpy as np
from numpy import ndarray as ndarray
import matplotlib.pyplot as plt
from random import randint

from utils.constants import NUM_ASSETS, NUM_FEATURES
from data_management.data_manager import PriceHistory
from utils.tools import train_test_split

class PortfolioManager():
    PRICE_TYPE = "ratio"

    def __init__(self, args, commission_rate:float=0.025, min_history: Optional[int] = None, flag="train", frequency: int = 1, **kwargs) -> None:
        self.frequency = frequency
        self.min_history = min_history or 0
        self.args = args

        self._set_dates(flag=flag)
        if not commission_rate:
          self.commission_rate_selling : float = args.commission_rate_purchasing
          self.commission_rate_purchasing : float  = args.commission_rate_selling
        else:
          self.commission_rate_selling = commission_rate
        self.state_space = PriceHistory(
            args,
            num_periods=args.seq_len,
            granularity=args.granularity,
            start_date=self.start_date,
            end_date=self.end_date,
            scale=False)
        
        self.start_value: int = 10000
        self.fee: float = 0.0025
        self.rf_rate = 0.0

        self.initial_weights = NUM_ASSETS * [1 / NUM_ASSETS]
        self.__set_X()


    def _set_dates(self, flag) -> None:
        start_date_train, end_date_train, start_date_test, end_date_test = train_test_split(self.args.ratio, self.args.granularity, self.args.start_date, self.args.end_date)
        self.start_date = start_date_train if flag=="train" else start_date_test
        self.end_date = end_date_test if flag=="test" else end_date_train

 
    def __set_X(self) -> None:
      self.X = self.state_space.filled_feature_matrices[0]
      self.X = self.X.rename(columns={"time": "date"}).set_index("date")
      self.X.columns.name = "Symbols"
      self.ratio = self._convert_prices(self.X, "ratio")


    def init_weights(self, columns):
        """Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(len(columns))


    def init_step(self, X):
        """Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """
        pass


    def step(self, x, last_b, history=None):
        """Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError("Subclass must implement this!")


    def weights(self, X, min_history=None):
        """Return weights. Call step method to update portfolio sequentially."""
        min_history = self.min_history if min_history is None else min_history
        # init
        B = X.copy() * 0.0
        last_b = self.init_weights(X.columns)
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)
        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.iloc[t] = last_b
            # keep initial weights for min_history
            if t < min_history:
                continue
            # trade each `frequency` periods
            if (t - min_history) % self.frequency != 0:
                continue
            # predict for t+1
            history = X.iloc[: t + 1]
            last_b = self.step(x, last_b, history)
            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))
        return B


    def run(self, P=None) -> pd.DataFrame:
        """Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
        """
        P = P if isinstance(P, pd.DataFrame) else self.X
        # convert prices to proper format
        X = self._convert_prices(P, self.PRICE_TYPE)
        # get weights
        B = self.weights(X)
        # cast to dataframe if weights return numpy array
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)
        return B
    

    def _convert_prices(self, S, method):
        """Convert prices to format suitable for weight or step function.
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == "raw":
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.items():
                if s.first_valid_index() is not None:
                    init_val = s.loc[s.first_valid_index()]
                    r[name] = s / init_val
            return pd.DataFrame(r)
        elif method == "absolute":
            return S

        elif method in ("ratio", "log"):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method="ffill")
            for name, s in X.iteritems():
                if s.first_valid_index() is not None:
                    X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.0
            return np.log(X) if method == "log" else X


    def calculate_returns(self, B=None):
      B = B if isinstance(B, pd.DataFrame) else self.run()
      # calculate return for individual stocks
      X = self._convert_prices(self.X, "ratio")
      r = (X - 1) * B #X "ratio"
      self.asset_r = r + 1
      self.r = r.sum(axis=1) + 1

      # stock went bankrupt
      self.r[self.r < 0] = 0.0

      # add risk-free asset
      self.r -= (B.sum(axis=1) - 1) #* self.rf_rate / self.freq()
      # add fees
      self.fees = self.to_rebalance(B, X).abs() * self.fee
      self.asset_r -= self.fees
      self.r -= self.fees.sum(axis=1)

      self.r = np.maximum(self.r, 1e-10)
      self.r_log = np.log(self.r)
      return self.r


    def sharpe(self, r=None, rf_rate=0.0, alpha=0.0, freq="daily", sd_factor=1.0, w=None):
      r = r if isinstance(r, pd.Series) else self.calculate_returns()
      if freq == "hourly":
        freq = 60 * 60 / self.state_space.granularity
      elif freq == "daily":
        freq = 24 * 60 * 60 / self.state_space.granularity
      elif freq == "monthly":
        freq = 30 * 24 * 60 * 60 / self.state_space.granularity
      # adjust rf rate by frequency
      #rf = rf_rate / freq
      # subtract risk-free rate
      #r = _sub_rf(r, rf)
      # freq return and sd
      if w is None:
          mu = r.mean()
          sd = r.std()
      else:
          mu = (r * w).sum() / w.sum() #w_avg(r, w)
          sd = np.sqrt(np.maximum(0, (r**2 * w).sum() / w.sum() - mu ** 2))#w_std(r, w)
      mu = mu * freq
      sd = sd * np.sqrt(freq)
      sh = mu / (sd + alpha) ** sd_factor
      #if isinstance(sh, float):
      #    if sh == np.inf:
      #        return np.inf * np.sign(mu - rf ** (1.0 / freq))
      #else:
      #    pass
          # sh[sh == np.inf] *= np.sign(mu - rf**(1./freq))
      self._sharpe = sh
      return sh


    def to_rebalance(self, B, X):
        """
        :param X: price relatives (1 + r)
        """
        # equity increase
        E = (B * (X - 1)).sum(axis=1) + 1
        X = X.copy()
        # calculate new value of assets and normalize by new equity to get
        # weights for tomorrow
        hold_B = (B * X).div(E, axis=0)
        return B - hold_B.shift(1)


    def plot_portfolio_value(self, r=None):
      r = r if r.any() else self.calculate_returns()
      temp = self.start_value 
      portfolio_values = []
      for return_ in list(r):
        portfolio_values.append(temp)
        temp *= return_
      plt.plot(portfolio_values, label=self.name)    
      #x r.index


    def plot_portfolio_weights(self, B=None):
      B = B if isinstance(B, pd.DataFrame) else self.run()
      B =  B.values
      plt.figure(figsize=(20, 5), dpi=80)
      plt.title("Portfolio Wheigts")
      plt.grid(b=None, which='major', axis='y', linestyle='--')

      for i in range(NUM_ASSETS):
        coin = self.state_space.coins[i].split("-")[0]
        #plt.plot(np.array(env.action_history)[1:, i], label=coin)
        plt.plot(B[:300, i], label=coin)
      plt.legend()
      plt.show()
