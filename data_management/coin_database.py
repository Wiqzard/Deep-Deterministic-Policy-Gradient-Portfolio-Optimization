import sqlite3
import requests
import pandas as pd
import json
import time
from numpy import ndarray as ndarray
from datetime import datetime, timedelta
from random import randint
import random
from tqdm import tqdm
import sqlite3


from utils.tools import get_date_minutes, logger
from utils.constants import *


class CoinDatabase:
    def __init__(self, args):
        self.__storage_period = FIVE_MINUTES
        self.features = FEATURES
        self.coins = COINS
        self.database_path = args.database_path

    def create_table(self, coin: str):
        with sqlite3.connect(self.database_path) as connection:
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
            self.create_table(coin)

    def check_tables(self) -> tuple:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return cursor.fetchall()

    def get_column_names(self, table_name: str) -> list:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM '{table_name}'")
            return list(map(lambda x: x[0], cursor.description))

    def fill_table(self, coin_ticker: str, granularity, start_date, end_date) -> None:
        with sqlite3.connect(self.database_path) as connection:
            table_name = f"{coin_ticker}-History"
            query = f'SELECT MIN(time) AS min_date, MAX(time) AS max_date FROM "{table_name}"'
            existing_data = pd.read_sql(query, connection)
            fill_new = (
                existing_data["min_date"].isnull().any()
                and existing_data["max_date"].isnull().any()
            )
            case1, case2 = 0, 0
            if fill_new:
                case1, case2 = 1, 1
                data_before = self.retrieve_data(
                    ticker=coin_ticker,
                    granularity=granularity,
                    start_date=start_date,
                    end_date=end_date,
                )
                data_after = None

            else:
                min_date = datetime.fromtimestamp(
                    existing_data["min_date"].iloc[0] * 60
                )
                max_date = datetime.fromtimestamp(
                    existing_data["max_date"].iloc[0] * 60
                )

                start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
                end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")
                if start_date_datetime < min_date and end_date_datetime > min_date:
                    min_date = min_date.strftime("%Y-%m-%d-%H-%M")
                    data_before = self.retrieve_data(
                        ticker=coin_ticker,
                        granularity=granularity,
                        start_date=start_date,
                        end_date=min_date,
                    )
                elif start_date_datetime < min_date and end_date_datetime < min_date:
                    min_date = min_date.strftime("%Y-%m-%d-%H-%M")
                    data_before = self.retrieve_data(
                        ticker=coin_ticker,
                        granularity=granularity,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    case1 = 1
                else:
                    data_before = None
                if end_date_datetime > max_date and start_date_datetime < max_date:
                    max_date = max_date.strftime("%Y-%m-%d-%H-%M")
                    data_after = self.retrieve_data(
                        ticker=coin_ticker,
                        granularity=granularity,
                        start_date=max_date,
                        end_date=end_date,
                    )
                elif end_date_datetime > max_date and start_date_datetime > max_date:
                    max_date = max_date.strftime("%Y-%m-%d-%H-%M")
                    data_after = self.retrieve_data(
                        ticker=coin_ticker,
                        granularity=granularity,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    case2 = 1
                else:
                    data_after = None

            query = f'INSERT INTO "{table_name}" (time, low, high, open, close, volume) VALUES (?, ?, ?, ?, ?, ?)'

            for data in [data_before, data_after]:
                if data is not None:
                    start_date_ = start_date
                    end_date_ = end_date
                    if data is data_before:
                        if case1 == 0:
                            end_date_ = min_date
                    elif data is data_after:
                        if case2 == 0:
                            start_date_ = max_date
                    for i, row in data.iterrows():
                        try:
                            connection.execute(
                                query,
                                (
                                    row["time"],
                                    row["low"],
                                    row["high"],
                                    row["open"],
                                    row["close"],
                                    row["volume"],
                                ),
                            )
                        except Exception as e:
                            logger.error(f"Error inserting data into table: {e}")
                            return

                    logger.info(
                        f"Inserting data for period from {start_date_} to {end_date_} into table {table_name}"
                    )

            connection.commit()

    def fill_all_tables(self, granularity, start_date, end_date) -> None:
        for coin in self.coins:
            self.fill_table(coin, granularity, start_date, end_date)

    def retrieve_data(
        self, ticker: str, granularity: int, start_date: str, end_date: str
    ) -> pd.DataFrame:  # sourcery skip: assign-if-exp, extract-method, inline-immediately-returned-variable

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d-%H-%M")

        start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
        end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")

        request_volume = (
            abs((start_date_datetime - end_date_datetime).total_seconds()) / granularity
        )
        if request_volume <= 300:
            data = self.retrieve_data_single_request(
                ticker, start_date_datetime, end_date_datetime, granularity
            )
        else:
            data = self.retrieve_data_multiple_requests(
                ticker, start_date_datetime, request_volume, granularity
            )
        data.columns = ["time", "low", "high", "open", "close", "volume"]
        data["time"] = data["time"] / 60  # s
        data["time"] = data["time"].astype(int)
        if data.empty:
            raise pd.errors.EmptyDataError("No data fetched.")
        return data

    def retrieve_data_single_request(self, ticker, start_date, end_date, granularity):
        response = requests.get(
            "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                ticker, start_date, end_date, granularity
            )
        )
        if response.status_code in [200, 201, 202, 203, 204]:
            data = pd.DataFrame(json.loads(response.text))
        else:
            raise ValueError(
                f"Received non-success response code: {response.status_code}"
            )
        return data

    def retrieve_data_multiple_requests(
        self, ticker, start_date_datetime, request_volume, granularity
    ):
        max_per_mssg = 300
        logger.info(f"Retrieve history for {ticker}")
        data = pd.DataFrame()
        for i in tqdm(
            range(int(request_volume / max_per_mssg) + 1), position=0, leave=True
        ):
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
            if response.status_code not in [200, 201, 202, 203, 204]:
                raise ValueError(
                    f"Received non-success response code: {response.status_code}"
                )

            dataset = pd.DataFrame(json.loads(response.text))
            if not dataset.empty:
                data = pd.concat([data, dataset], ignore_index=True, sort=False)
                time.sleep(2 * random.random())
        return data

    def get_coin_data(
        self, coin: str, granularity, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start_date_minutes = get_date_minutes(start_date)
        end_date_minutes = get_date_minutes(end_date) if end_date is not None else None

        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()

            table_name = f"{coin}-History"
            features_str = '"low", "high", "open", "close", "volume" '
            sql = f'SELECT DISTINCT "time", {features_str} FROM "{table_name}" WHERE time>={start_date_minutes}'
            if end_date_minutes is not None:
                sql += f" and time<={str(end_date_minutes)}"
            sql += f" and time%{int(granularity / 60)}=0"

            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=["time"] + self.features)

            if df.empty:
                raise ValueError("No data found for the specified parameters.")

            df = df.sort_values("time")
            df["time"] = pd.to_datetime(df["time"], unit="m")
            df = df.reset_index(drop=True)

            first_date = df["time"].iloc[0]
            last_date = df["time"].iloc[-1]
            start_date_str = first_date.strftime("%Y-%m-%d %H:%M:%S")
            end_date_str = last_date.strftime("%Y-%m-%d %H:%M:%S")
            s_date_dt = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
            e_date_dt = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")

            if s_date_dt < first_date and e_date_dt > last_date:
                logger.info(
                    f"Could only fetch data from {start_date_str} to {end_date_str}. Consider to fill the database."
                )
            elif s_date_dt < first_date and e_date_dt < last_date:
                logger.info(
                    f"Could only fetch data from {start_date_str} to {end_date}. Consider to fill the database."
                )
            elif s_date_dt > first_date and e_date_dt > last_date:
                logger.info(
                    f"Could only fetch data from {start_date} to {end_date_str}. Consider to fill the database."
                )
            elif s_date_dt > first_date and e_date_dt < last_date:
                logger.info(
                    f"Could only fetch data from {start_date} to {end_date}. Consider to fill the database."
                )

            return df
