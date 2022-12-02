from datetime import datetime


def get_date_minutes(date: str) -> int:
    date_datetime = datetime.strptime(date, "%Y-%m-%d-%H-%M")
    time_delt = date_datetime - datetime(1970, 1, 1)
    return int(time_delt.total_seconds() / 60)