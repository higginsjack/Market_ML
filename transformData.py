import pandas as pd
from datetime import datetime
import os
import time

"""
Momentum:
    * If we try use this in a day trading context we should try to find points where we see a 5%-10% (maybe smaller) increase and find the beginning of that
        * Basically local minimum identification
    * Over what period of time?, 
        * Within an hour, 2 hours, etc.
        * Should growth be sustained or spike?
        * Should surrounding points be similarly incentivized?
    * 
"""

def growthPeriodsFlat(df, time_window, percentage_change):
    """
    Params:
    df: Close and datetimes needed
    time_windows: (list of integers) that correspond to periods of time
    percentage_change: (list of integers) what is minimum percentage changes we want to identify

    Returns:
    dataframe containing start times of growth

    This data will be binary, and not incentivize bigger returns, shorter time periods, or sustained growth
    """

    