from datetime import datetime, timedelta
import logging
from typing import Tuple
import multiprocessing
import threading
import sys 


def get_date_minutes(date: str) -> int:
    date_datetime = datetime.strptime(date, "%Y-%m-%d-%H-%M")
    time_delt = date_datetime - datetime(1970, 1, 1)
    return int(time_delt.total_seconds() / 60)


logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)




def train_test_split(ratio:float, granularity:int,  start_date: str, end_date: str) -> Tuple[str, str, str, str]:
    date_format = "%Y-%m-%d-%H-%M"
    try:
        start_date = datetime.strptime(start_date, date_format)
    except ValueError as e:
        raise ValueError("Invalid start date format. Please use the format yyyy-mm-dd-hh-mm") from e

    minute = round(start_date.minute / granularity) * granularity
    start_date = start_date.replace(minute=minute)

    if end_date is None:
        now = datetime.now()
        minute = round(now.minute / granularity) * granularity 
        end_date = now.replace(minute=minute)
    else:
        try:
            end_date = datetime.strptime(end_date, date_format)
        except ValueError as exc:
            raise ValueError("Invalid end date format. Please use the format yyyy-mm-dd-hh-mm") from exc

        minute = round(end_date.minute / granularity) * granularity
        end_date = end_date.replace(minute=minute)

    diff_minutes = (end_date - start_date).total_seconds() / 60
    minutes_into_period = diff_minutes * ratio
    minutes_into_period = round(minutes_into_period / (granularity / 60)) * granularity/60
    date1 = start_date + timedelta(minutes=minutes_into_period)
    date2 = date1 + timedelta(minutes=granularity / 60)

    start_date_train = start_date.strftime("%Y-%m-%d-%H-%M")
    end_date_train= date1.strftime("%Y-%m-%d-%H-%M")
    start_date_test = date2.strftime("%Y-%m-%d-%H-%M")
    end_date_test = end_date.strftime("%Y-%m-%d-%H-%M")

    return start_date_train, end_date_train, start_date_test, end_date_test


def count_granularity_intervals(start_date:str, end_date:str, granularity:int):
    start_date = datetime.strptime(start_date, "%Y-%m-%d-%H-%M")
    end_date = datetime.strptime(end_date, "%Y-%m-%d-%H-%M")
    interval = end_date - start_date
    return int(interval.total_seconds() // granularity)


#def check_user_input(abort:bool) -> None:
#    while not abort:
#        input_str = input("Enter 'abort' to terminate loop: ")
#        if input_str.lower() == "abort":
#            abort = True
#def check_user_input(abort:threading.Event) -> None:
#    while not abort.is_set():
#        input_str = input("Enter 't' to terminate loop: \n")
#        if input_str.lower() == "t":
#            abort.set()        
#            logger.info("Train Loop will terminate after current episode.")    
#def check_user_input(abort:multiprocessing.Value) -> None:
#    while not abort.value:
#        input_str = input("Enter 'abort' to terminate loop: \n")
#        if input_str.lower() == "abort":
#            abort.value = True

