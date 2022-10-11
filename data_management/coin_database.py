import datetime
import logging
import sqlite3
from os import path
import pandas as pd
import requests

import numpy as np
from numpy import ndarray as ndarray
import json
import time
import datetime
from datetime import datetime, timedelta
from random import randint
from tqdm.auto import tqdm
from data_management.utils import get_date_minutes


DATABASE_DIR = "coin_history.db"
# CONFIG_FILE_DIR = 'net_config.json'
# LAMBDA = 1e-4  # lambda in loss function 5 in training
# About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
# trading table name
TABLE_NAME = "test"


"""
IDEA:
    -> own table named "coin_history" for each individual coin
        -> table has columns "date", "feature1", "feature2", "feature3"
    

    Do: retreive data implement functions from utils to make it more compact, also modularize
"""


class CoinDatabase:
    def __init__(self):
        self.__storage_period = FIVE_MINUTES
        self.features = ["low", "high", "open", "close", "volume"]
        self.coins = self.coins = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "SOL-USD",
            "DOGE-USD",
            "DOT-USD",
            "DAI-USD",
            "SHIB-USD",
        ]

    def create_table(self, coin: str):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS "{coin}-History" (time INTEGER,'
                "low FLOAT, high FLOAT, "
                " open FLOAT, close FLOAT, volume FLOAT, "
                "PRIMARY KEY (time));".format(coin=coin)
            )
            connection.commit()

    def create_all_tables(self):
        for coin in self.coins:
            # coin = coin.split("-")[0]
            self.create_table(coin)

    def check_tables(self) -> tuple:
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return cursor.fetchall()

    def get_column_names(self, table_name: str) -> list:
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM '{table_name}'")
            return list(map(lambda x: x[0], cursor.description))

    def fill_table(self, coin_ticker: str, granularity, start_date, end_date):
        data = self.retrieve_data(
            ticker=coin_ticker,
            granularity=granularity,
            start_date=start_date,
            end_date=end_date,
        )
        with sqlite3.connect(DATABASE_DIR) as connection:
            table_name = f"{coin_ticker}-History"
            data.to_sql(table_name, connection, if_exists="replace", index=False)
            connection.commit()

    def fill_all_tables(self, granularity, start_date, end_date) -> None:
        for coin in self.coins:
            self.fill_table(coin, granularity, start_date, end_date)

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
            pbar.close()
            data.columns = ["time", "low", "high", "open", "close", "volume"]
            data["time"] = data["time"] / 60  # s
            data["time"] = data["time"].astype(int)
            # data["time"] = pd.to_datetime(data["time"], unit="m")
            # data = data[data["time"].between(start_date_datetime, end_date_datetime)]
            # data.set_index("time", drop=True, inplace=True)
            # data.sort_index(ascending=True, inplace=True)
            data.drop_duplicates(subset=None, keep="first", inplace=True)
            return data

    def get_coin_data(
        self, coin: str, granularity, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start_date_minutes = get_date_minutes(start_date)

        if end_date is not None:
            end_date_minutes = get_date_minutes(end_date)
            end_sql = f"and time<={str(end_date_minutes)}"
        else:
            end_sql = ""

        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()

            table_name = f"{coin}-History"
            features_str = ",".join(map(str, self.features))
            features_str = '"low", "high", "open", "close", "volume" '
            sql = (
                'SELECT DISTINCT "time",'
                + features_str
                + """ FROM "{table_name}"
             WHERE time>={start_date} {end_sql}
            and time%{granularity}=0""".format(
                    table_name=table_name,
                    start_date=start_date_minutes,
                    end_sql=end_sql,
                    granularity=int(granularity / 60),
                )
            )
            # print(sql)
            cursor.execute(sql)

            df = pd.DataFrame(cursor.fetchall(), columns=["time"] + self.features)
            df.set_index("time", drop=True, inplace=True)
            df.sort_index(ascending=True, inplace=True)
            df.head()
            return df
