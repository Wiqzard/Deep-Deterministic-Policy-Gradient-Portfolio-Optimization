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
from datetime import datetime, timedelta
from random import randint
from tqdm.auto import tqdm


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
    


"""
class CoinDatabase:
    def __init__(self):
        self.__storage_period = FIVE_MINUTES
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



    def create_table(self, coin):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS "{coin}"-History (date INTEGER,'
                "low FLOAT, high FLOAT, "
                " open FLOAT, close FLOAT, volume FLOAT, "
                "PRIMARY KEY (date));".format(coin = coin)
            )
            connection.commit()

    def create_all_tables(self):
        for coin in self.coins:
            coin = coin.split("-")[0]
            self.create_table(coin)

    def fill_table(self, idx):

        with sqlite3.connect(DATABASE_DIR) as connection:

    def retrieve_data(
        self, ticker: str, granularity: int, start_date: str, end_date: str) -> pd.DataFrame:

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

class CoinHistory:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(
        self, coin_number, end, volume_average_days=1, volume_forward=0, online=True
    ):
        self.initialize_db()
        self.__storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins


    def get_global_data_matrix(self, start, end, period=300, features=("close",)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values

    def get_global_panel(self, start, end, period=300, features=("close",)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start - (start % period))
        end = int(end - (end % period))
        coins = self.select_coins(
            start=end - self.__volume_forward - self.__volume_average_days * DAY,
            end=end - self.__volume_forward,
        )
        self.__coins = coins
        for coin in coins:
            self.update_data(start, end, coin)

        if len(coins) != self._coin_number:
            raise ValueError(
                "the length of selected coins %d is not equal to expected %d"
                % (len(coins), self._coin_number)
            )

        logging.info(f"feature type list is {str(features)}")
        self.__checkperiod(period)

        time_index = pd.to_datetime(list(range(start, end + 1, period)), unit="s")
        panel = pd.Panel(
            items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32
        )

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for coin in coins:
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = (
                            "SELECT date+300 AS date_norm, close FROM History WHERE"
                            " date_norm>={start} and date_norm<={end}"
                            ' and date_norm%{period}=0 and coin="{coin}"'.format(
                                start=start, end=end, period=period, coin=coin
                            )
                        )
                    elif feature == "high":
                        sql = (
                            "SELECT date_norm, MAX(high)"
                            + " FROM (SELECT date+{period}-(date%{period})"
                            " AS date_norm, high, coin FROM History)"
                            ' WHERE date_norm>={start} and date_norm<={end} and coin="{coin}"'
                            " GROUP BY date_norm".format(
                                period=period, start=start, end=end, coin=coin
                            )
                        )
                    elif feature == "low":
                        sql = (
                            "SELECT date_norm, MIN(low)"
                            + " FROM (SELECT date+{period}-(date%{period})"
                            " AS date_norm, low, coin FROM History)"
                            ' WHERE date_norm>={start} and date_norm<={end} and coin="{coin}"'
                            " GROUP BY date_norm".format(
                                period=period, start=start, end=end, coin=coin
                            )
                        )
                    elif feature == "open":
                        sql = (
                            "SELECT date+{period} AS date_norm, open FROM History WHERE"
                            " date_norm>={start} and date_norm<={end}"
                            ' and date_norm%{period}=0 and coin="{coin}"'.format(
                                start=start, end=end, period=period, coin=coin
                            )
                        )
                    elif feature == "volume":
                        sql = (
                            "SELECT date_norm, SUM(volume)"
                            + " FROM (SELECT date+{period}-(date%{period}) "
                            "AS date_norm, volume, coin FROM History)"
                            ' WHERE date_norm>={start} and date_norm<={end} and coin="{coin}"'
                            " GROUP BY date_norm".format(
                                period=period, start=start, end=end, coin=coin
                            )
                        )
                    else:
                        msg = f"The feature {feature} is not supported"
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(
                        sql,
                        con=connection,
                        parse_dates=["date_norm"],
                        index_col="date_norm",
                    )
                    panel.loc[feature, coin, serial_data.index] = serial_data.squeeze()
                    panel = panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()
        return panel

    # select top coin_number of coins by volume from start to end
    def select_coins(self, start, end):
        if not self._online:
            logging.info(
                f"select coins offline from {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}"
            )

            connection = sqlite3.connect(DATABASE_DIR)
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT coin,SUM(volume) AS total_volume FROM History WHERE"
                    " date>=? and date<=? GROUP BY coin"
                    " ORDER BY total_volume DESC LIMIT ?;",
                    (int(start), int(end), self._coin_number),
                )
                coins_tuples = cursor.fetchall()

                if len(coins_tuples) != self._coin_number:
                    logging.error("the sqlite error happend")
            finally:
                connection.commit()
                connection.close()
            coins = [tuple[0] for tuple in coins_tuples]
        else:
            coins = list(self._coin_list.topNVolume(n=self._coin_number).index)
        logging.debug(f"Selected coins are: {coins}")
        return coins

    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError("peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day")

    # add new history data into the database
    def update_data(self, start, end, coin):
        # sourcery skip: raise-specific-error
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute(
                "SELECT MIN(date) FROM History WHERE coin=?;", (coin,)
            ).fetchall()[0][0]
            max_date = cursor.execute(
                "SELECT MAX(date) FROM History WHERE coin=?;", (coin,)
            ).fetchall()[0][0]

            if min_date is None or max_date is None:
                self.__fill_data(start, end, coin, cursor)
            else:
                if max_date + 10 * self.__storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self.__fill_data(
                        max_date + self.__storage_period, end, coin, cursor
                    )
                if min_date > start and self._online:
                    self.__fill_data(
                        start, min_date - self.__storage_period - 1, coin, cursor
                    )

                # if there is no data
        finally:
            connection.commit()
            connection.close()

    def __fill_data(self, start, end, coin, cursor):
        duration = 7819200  # three months
        bk_start = start
        for bk_end in range(bk_start + duration - 1, end, duration):
            self.__fill_part_data(bk_start, bk_end, coin, cursor)
            bk_start += duration
        if bk_start < end:
            self.__fill_part_data(bk_start, end, coin, cursor)

    def __fill_part_data(self, start, end, coin, cursor):
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, "pair"],
            start=start,
            end=end,
            period=self.__storage_period,
        )
        logging.info(
            f"fill {coin} data from {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}"
        )

        for c in chart:
            if c["date"] > 0:
                if c["weightedAverage"] == 0:
                    weightedAverage = c["close"]
                else:
                    weightedAverage = c["weightedAverage"]

                # NOTE here the USDT is in reversed order
                if "reversed_" in coin:
                    cursor.execute(
                        "INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            c["date"],
                            coin,
                            1.0 / c["low"],
                            1.0 / c["high"],
                            1.0 / c["open"],
                            1.0 / c["close"],
                            c["quoteVolume"],
                            c["volume"],
                            1.0 / weightedAverage,
                        ),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            c["date"],
                            coin,
                            c["high"],
                            c["low"],
                            c["open"],
                            c["close"],
                            c["volume"],
                            c["quoteVolume"],
                            weightedAverage,
                        ),
                    )
