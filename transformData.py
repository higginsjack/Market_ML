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
def unix_to_datetime(unix):
    return datetime.fromtimestamp(unix)

def find_nearest_date(df, target_date):
    """
    Find the index of the nearest date in the DataFrame to the target date.
    """
    nearest_index = (df["Timestamp"] - target_date).abs().idxmin()
    return nearest_index

def growthDF(df, time_window):
    """
    Params:
    df: DataFrame containing 'Unix' and 'Close' columns
    time_window: Dictionary of time windows (in seconds) for growth calculation

    Returns:
    growth_df: DataFrame containing start times and growth percentages for different time windows
    """
    growth_df = pd.DataFrame(columns=["Unix", "Timestamp"] + list(time_window.keys()))
    
    data = {
        "unix": [],
        "tstamp": [],
    }
    
    for window in time_window:
        data[window] = []
    
    for idx, row in df.iterrows():
        data["unix"].append(row["Unix"])
        data["tstamp"].append(unix_to_datetime(row["Unix"]))
        if idx % 1000 == 0:
            print(unix_to_datetime(row["Unix"]))

        for window in time_window:
            target_date = unix_to_datetime(row["Unix"] + window)
            nearest_index = find_nearest_date(df, target_date)
            nearest_row = df.loc[nearest_index]
            nearest_date = nearest_row["Timestamp"]
            
            if abs((target_date - nearest_date).total_seconds()) < (0.1 * window):
                price_change = (nearest_row["Close"] - row["Close"]) / row["Close"]
                data[window].append(price_change)
            else:
                data[window].append(-1)
    
    growth_df["Unix"] = data["unix"]
    growth_df["Timestamp"] = data["tstamp"]
    
    for window in time_window:
        growth_df[window] = data[window]
    
    return growth_df

# Example usage
time_window = {
    15*60: "15min",
    30*60: "30min",
    60*60: "1hr",
    120*60: "2hr",
    180*60: "3hr"
}

ticker = 'AMZN'
df = pd.read_csv('data/' + ticker +'/aggregate/' + ticker +'_agg.csv')
df = df[["Unix","Timestamp","Close"]]
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

growth_df = growthDF(df, time_window)
growth_df.to_csv("data/" + ticker + "/aggregate/" + ticker + "_growth.csv")
print(growth_df)