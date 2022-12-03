import sqlite3
import requests
import pandas as pd
import json
import time
from numpy import ndarray as ndarray
from datetime import datetime, timedelta
from random import randint
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

    def fill_table(self, coin_ticker: str, granularity, start_date, end_date):
        data = self.retrieve_data(
            ticker=coin_ticker,
            granularity=granularity,
            start_date=start_date,
            end_date=end_date,
        )
        try:
            with sqlite3.connect(self.database_path) as connection:
                table_name = f"{coin_ticker}-History"
                data.to_sql(table_name, connection, if_exists="replace", index=False)
                connection.commit()
        except ValueError as e:
            raise ValueError("Retrieved data is None") from e


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
            data = self.retrieve_data_single_request(ticker, start_date_datetime, end_date_datetime, granularity)
        else:
            data = self.retrieve_data_multiple_requests(ticker, start_date_datetime, request_volume, granularity)
        
        data.columns = ["time", "low", "high", "open", "close", "volume"]
        data["time"] = data["time"] / 60  # s
        data["time"] = data["time"].astype(int)
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
            raise ValueError(f"Received non-success response code: {response.status_code}")
        return data


    def retrieve_data_multiple_requests(self, ticker, start_date_datetime, request_volume, granularity):
        max_per_mssg = 300
        logger.info(f"Retrieve history for {ticker}")
        data = pd.DataFrame()
        for i in tqdm(range(int(request_volume / max_per_mssg) + 1), position=0, leave=True):
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
                raise ValueError(f"Received non-success response code: {response.status_code}")

            dataset = pd.DataFrame(json.loads(response.text))
            if not dataset.empty:
                data = pd.concat([data, dataset], ignore_index=True, sort=False)
                time.sleep(randint(0, 1))
        return data


#    def retrieve_data(
#        self, ticker: str, granularity: int, start_date: str, end_date: str
#    ) -> pd.DataFrame:  # sourcery skip: extract-method
#
#        if end_date is None:
#            end_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
#
#        start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
#        end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")
#
#        request_volume = (
#            abs((start_date_datetime - end_date_datetime).total_seconds()) / granularity
#        )
#        if request_volume <= 300:
#            response = requests.get(
#                "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
#                    ticker, start_date, end_date, granularity
#                )
#            )
#        else:
#            max_per_mssg = 300
#            logger.info(f"Retrieve history for {ticker}")
#            data = pd.DataFrame()
#            for i in tqdm(range(int(request_volume / max_per_mssg) + 1), position=0, leave=True):
#                provisional_start = start_date_datetime + timedelta(
#                    0, i * (granularity * max_per_mssg)
#                )
#                provisional_end = start_date_datetime + timedelta(
#                    0, (i + 1) * (granularity * max_per_mssg)
#                )
#                response = requests.get(
#                    "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
#                        ticker, provisional_start, provisional_end, granularity
#                    )
#                )
#                if response.status_code in [200, 201, 202, 203, 204]:
#                    dataset = pd.DataFrame(json.loads(response.text))
#                    if not dataset.empty:
#                        data = pd.concat([data, dataset], ignore_index=True, sort=False)
#                        time.sleep(randint(0, 2))
#                else:
#                    print("Something went wrong")
#            data.columns = ["time", "low", "high", "open", "close", "volume"]
#            data["time"] = data["time"] / 60  # s
#            data["time"] = data["time"].astype(int)
#            return data

    #def get_coin_data(
    #    self, coin: str, granularity, start_date: str, end_date: str
    #) -> pd.DataFrame:
    #    start_date_minutes = get_date_minutes(start_date)

    #    if end_date is not None:
    #        end_date_minutes = get_date_minutes(end_date)
    #        end_sql = f"and time<={str(end_date_minutes)}"
    #    else:
    #        end_sql = ""

    #    with sqlite3.connect(self.database_path) as connection:
    #        cursor = connection.cursor()

    #        table_name = f"{coin}-History"
    #        features_str = ",".join(map(str, self.features))
    #        features_str = '"low", "high", "open", "close", "volume" '
    #        sql = (
    #            'SELECT DISTINCT "time",'
    #            + features_str
    #            + """ FROM "{table_name}"
    #         WHERE time>={start_date} {end_sql}
    #        and time%{granularity}=0""".format(
    #                table_name=table_name,
    #                start_date=start_date_minutes,
    #                end_sql=end_sql,
    #                granularity=int(granularity / 60),
    #            )
    #        )
    #        cursor.execute(sql)
    #        df = pd.DataFrame(cursor.fetchall(), columns=["time"] + self.features)
    #        df = df.sort_values("time")
    #        df["time"] = pd.to_datetime(df["time"], unit="m")
    #        df = df.reset_index(drop=True)
    #        return df
    def get_coin_data(self, coin: str, granularity, start_date: str, end_date: str) -> pd.DataFrame:
        start_date_minutes = get_date_minutes(start_date)

        end_date_minutes = get_date_minutes(end_date) if end_date is not None else None

        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()

            table_name = f"{coin}-History"
            features_str = '"low", "high", "open", "close", "volume" '
            sql = f'SELECT DISTINCT "time", {features_str} FROM "{table_name}" WHERE time>={start_date_minutes}'
            if end_date_minutes is not None:
                sql += f" and time<={str(end_date_minutes)}"
            sql += f' and time%{int(granularity / 60)}=0'

            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=["time"] + self.features)
            if df.empty:
                raise ValueError("No data found for the specified parameters.")
            df = df.sort_values("time")
            df["time"] = pd.to_datetime(df["time"], unit="m")
            df = df.reset_index(drop=True)
            return df