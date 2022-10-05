import requests
import json
import time
from random import randint
import pandas as pd
import sys
from datetime import datetime, timedelta


tkr_response = requests.get("https://api.pro.coinbase.com/products")
if tkr_response.status_code in [200, 201, 202, 203, 204]:
    print("Connected to the CoinBase Pro API.")


def retrieve_data(
    ticker: str, granularity: int, start_date: str, end_date: str = None
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
                if i % 3 == 0:
                    print(
                        f"Data for chunk {i + 1} of {int(request_volume / max_per_mssg) + 1} extracted"
                    )

                dataset = pd.DataFrame(json.loads(response.text))
                if not dataset.empty:
                    data = data.append(dataset)
                    time.sleep(randint(0, 2))
            else:
                print("Something went wrong")
        data.columns = ["time", "low", "high", "open", "close", "volume"]
        data["time"] = pd.to_datetime(data["time"], unit="s")
        data = data[data["time"].between(start_date_datetime, end_date_datetime)]
        data.set_index("time", drop=True, inplace=True)
        data.sort_index(ascending=True, inplace=True)
        data.drop_duplicates(subset=None, keep="first", inplace=True)
        return data


ticker = "ATOM-USD"
granularity = 900
start_date = "2022-09-14-00-00"
data = retrieve_data(ticker, granularity, start_date)
# Stable coin as numeraire?
coins = [
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "ADA",
    "SOL",
    "DOGE",
    "DOT",
    "DAI",
    "SHIB",
    "TRX",
    "AVAX",
    "UNI",
    "WBTC",
    "LEO",
    "ETC",
    "ATOM",
    "LINK",
    "LTC",
    "FTT",
]
