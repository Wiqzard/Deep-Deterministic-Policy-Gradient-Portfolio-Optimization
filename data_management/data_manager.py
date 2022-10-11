import pandas as pd
import numpy as np
from numpy import ndarray as ndarray
import datetime
import requests
import json
import time
from datetime import datetime, timedelta
from random import randint
from tqdm.auto import tqdm
import logging
from data_management.coin_database import CoinDatabase


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


class PriceHistory:
    def __init__(
        self,
        num_features: int,
        num_periods: int,
        granularity: int = None,
        start_date: str = None,
        end_date: str = None,
        data_base=None,
    ):
        self.coins = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "SOL-USD",
            "DOGE-USD",
            "DOT-USD",
            "DAI-USD",
            "SHIB-USD",
        ]
        # "AVAX-USD",
        # "UNI-USD",
        # "WBTC-USD",
        # "ETC-USD",
        # "ATOM-USD",
        # "LINK-USD",
        # "LTC-USD",]
        self.num_features = num_features
        self.num_periods = num_periods
        self.num_assets = len(self.coins)
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.data_matrix = []  # np.empty(self.num_assets
        self.data_base = data_base

    def retrieve_data(
        self, ticker: str, granularity: int, start_date: str, end_date: str
    ) -> pd.DataFrame:

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d-%H-%M")

        start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
        end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")

        request_volume = (
            abs((start_date_datetime - end_date_datetime).total_seconds()) / granularity
        )
        if request_volume <= 300:
            response = requests.get(
                "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                    ticker, start_date, end_date, granularity
                )
            )
        else:
            max_per_mssg = 300
            logging.info(f"Retrieve history for {ticker}")
            pbar = tqdm(total=request_volume)
            data = pd.DataFrame()
            for i in range(int(request_volume / max_per_mssg) + 1):
                provisional_start = start_date_datetime + timedelta(
                    0, i * (granularity * max_per_mssg)
                )
                provisional_end = start_date_datetime + timedelta(
                    0, (i + 1) * (granularity * max_per_mssg)
                )
                response = requests.get(
                    "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                        ticker, provisional_start, provisional_end, granularity
                    )
                )
                if response.status_code in [200, 201, 202, 203, 204]:
                    pbar.update(max_per_mssg)

                    dataset = pd.DataFrame(json.loads(response.text))
                    if not dataset.empty:
                        data = pd.concat([data, dataset], ignore_index=True, sort=False)
                        time.sleep(randint(0, 2))
                else:
                    print("Something went wrong")
            # pbar.close()
            data.columns = ["time", "low", "high", "open", "close", "volume"]
            data["time"] = pd.to_datetime(data["time"], unit="s")
            data = data[data["time"].between(start_date_datetime, end_date_datetime)]
            data.set_index("time", drop=True, inplace=True)
            data.sort_index(ascending=True, inplace=True)
            data.drop_duplicates(subset=None, keep="first", inplace=True)
            return data

    # Implement error handling
    def set_data_matrix(
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

    def normalized_price_martix_asset(
        self, data_asset: pd.DataFrame, idx: int
    ) -> ndarray:
        data_asset = data_asset[["close", "high", "low"]].to_numpy().T
        temp_x_t = data_asset[:, idx - self.num_periods + 1 : idx + 1]
        V_x_t = temp_x_t / data_asset[:, idx, None]
        return V_x_t[:, :, np.newaxis]

    # Put in whole datamatrix [feature_number, num_data_points, number_assets] + idx
    def normalized_price_matrix(self, idx: int) -> ndarray:
        idx += self.num_periods - 1
        assert idx >= self.num_periods and idx <= self.data_matrix[0].shape[0]
        X_t = np.empty((self.num_features, self.num_periods, 1))

        for asset in range(self.num_assets):
            X_t = np.concatenate(
                (X_t, self.normalized_price_martix_asset(self.data_matrix[asset], idx)),
                axis=2,
            )
            # print(
            #    f"normprice: {self.normalized_price_martix_asset(self.data_matrix[asset], idx).shape}"
            # )
        # print(f"X_t: {X_t.shape}")
        X_t = X_t[:, :, 1:]
        # [feature_number, num_periods before t ascending, num_assets]
        # print(f"X_t: {X_t.shape}")

        return X_t
