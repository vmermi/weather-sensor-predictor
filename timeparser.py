import pandas as pd
import numpy as np
import datetime
from typing import List


class TimeParser:
    """
    Convert pandas.Series of string-like dates into an arrangement of seasons and slots.
    Seasons split the series into groups of days. Slots split each 24h day into groups
    of hours. The goal is to minimize the dimensionality of the data without losing too
    much information.

    Args:
        days_per_season (int, optional): The number of days in each season. Defaults to
            15.
        hours_per_slot (int | float, optional): The number of hours in each daily slot.
            Defaults to 2.
    """

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, days_per_season: int = 15, hours_per_slot: int | float = 2):
        if type(days_per_season) != int:
            raise TypeError("'days_per_season' must be an integer.")
        if days_per_season == 0:
            raise ValueError("'days_per_season' must not be zero.")
        if type(days_per_season) != int and type(days_per_season) != float:
            raise TypeError("'hours_per_slot' must be an integer/float.")
        if hours_per_slot == 0:
            raise ValueError("'hours_per_slot' must not be 0.")
        if 24 % hours_per_slot:
            raise ValueError("'hours_per_slot' must be a divisor of 24.")

        self.__days_per_season = days_per_season
        self.__hours_per_slot = hours_per_slot
        self.__slot_count = int(24 // self.__hours_per_slot)

        self.__start_datetime: datetime.datetime | None = None
        self.__end_datetime: datetime.datetime | None = None
        self.__season_count: int | None = None

    @property
    def start_date(self) -> datetime.date:
        if self.__start_datetime is None:
            raise RuntimeError("You should call the 'fit' method first.")
        return self.__start_datetime.date()

    @property
    def end_date(self) -> datetime.date:
        if self.__end_datetime is None:
            raise RuntimeError("You should call the 'fit' method first.")
        return self.__end_datetime.date()

    @property
    def start_datetime(self) -> datetime.datetime:
        if self.__start_datetime is None:
            raise RuntimeError("You should call the 'fit' method first.")
        return self.__start_datetime

    @property
    def end_datetime(self) -> datetime.datetime:
        if self.__end_datetime is None:
            raise RuntimeError("You should call the 'fit' method first.")
        return self.__end_datetime

    @property
    def slot_count(self) -> int:
        return self.__slot_count

    @property
    def season_count(self) -> int:
        if self.__season_count is None:
            raise RuntimeError("You should call the 'fit' method first.")
        return self.__season_count

    def fit(self, timeseries: pd.Series) -> None:
        """
        Compute the minimum and maximum to be used for later scaling.

        Args:
            timeseries (pd.Series): The timeseries used to compute the minimum and
                maximum
        """
        min_date = timeseries.min()
        max_date = timeseries.max()
        if min_date is None or max_date is None:
            raise ValueError("The timeseries is empty.")
        self.__start_datetime = self.str_to_datetime(min_date)
        self.__end_datetime = self.str_to_datetime(max_date)

        # Calculate the number of seasons needed for all data to fit
        assert (
            self.__days_per_season != 0
        ), "The number of days per season should not be zero"
        data_days = (self.end_date - self.start_date).days
        self.__season_count = data_days // self.__days_per_season
        if data_days % self.__days_per_season:
            self.__season_count += 1

    def transform(self, timeseries: pd.Series) -> pd.DataFrame:
        """
        Convert the timeseries into Seasons and time Slots. A pd.DataFrame will be
        returned, with Season and Slot columns, which are flags. Each column indicates
        if the observation belongs to that Season/Slot.

        Args:
            timeseries (pd.Series): Input data for the conversion

        Returns:
            pd.DataFrame: The converted output DataFrame
        """
        if self.__season_count is None:
            raise RuntimeError("You should call the 'fit' method first.")

        columns = [f"Season_{i}" for i in range(self.__season_count)] + [
            f"Slot_{i}" for i in range(self.__slot_count)
        ]

        timeseries = timeseries.apply(self.str_to_datetime)
        data = np.stack(list(timeseries.apply(self.two_hot_encode)))
        return pd.DataFrame(data, columns=columns)

    def fit_transform(self, timeseries: pd.Series) -> pd.DataFrame:
        """
        Fit to data, then transform them.

        Args:
            timeseries (pd.Series): Input data for the conversion

        Returns:
            pd.DataFrame: The converted output DataFrame
        """
        self.fit(timeseries)
        return self.transform(timeseries)

    def calculate_seasons(self, dt: datetime.datetime) -> List[bool]:
        """
        Calculates the season in which a datetime.datetime object belongs, in
        one-hot-encoded format.

        Args:
            dt (datetime.datetime): The datetime object for which the seasons list will
                be calculated

        Returns:
            List[bool]: The calculated one-hot-encoded season list
        """
        if self.start_date is None:
            raise RuntimeError("You should call the 'fit' method first.")
        season_index = int((dt.date() - self.start_date).days // self.__days_per_season)
        seasons = [False] * self.season_count
        seasons[season_index] = True
        return seasons

    def calculate_timeslots(self, dt: datetime.datetime) -> List[bool]:
        """
        Calculates the time slot in which a datetime.datetime object belongs, in
        one-hot-encoded format.

        Args:
            dt (datetime.datetime): The datetime object for which the time slots will be
                calculated

        Returns:
            List[bool]: The calculated one-hot-encoded time slots list
        """
        dt_in_minutes = dt.time().hour * 60 + dt.time().minute
        slot_index = int(dt_in_minutes // (self.__hours_per_slot * 60))
        slots = [False] * self.slot_count
        slots[slot_index] = True
        return slots

    def two_hot_encode(self, dt: datetime.datetime) -> List[bool]:
        """
        Encodes a datetime.datetime object into a list of booleans. The list contains
        exactly two true values corresponding to the season and time slot it belongs to.

        Args:
            dt (datetime.datetime): The datetime object that will be encoded

        Returns:
            List[bool]: The encoded datetime
        """
        return self.calculate_seasons(dt) + self.calculate_timeslots(dt)

    @classmethod
    def str_to_datetime(cls, string: str) -> datetime.datetime:
        """
        Converts a string into a datetime object.

        Args:
            string (str): The string containing the datetime

        Returns:
            datetime.datetime: The converted datetime
        """
        return datetime.datetime.strptime(string, cls.DATE_FORMAT)
