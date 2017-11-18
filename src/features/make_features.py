from datetime import datetime
import pandas as pd
import numpy as np
import os
from utils.general_utils import get_starting_and_end_time_per_trip
from sklearn.preprocessing import StandardScaler

def get_weekday_features(date_frame):
    """
    This method calculates the weekday for a given date column.
    :input date_frame: pandas DataFrame of the datetime objects.
    :return weekdays: pandas DataFrame of "Monday = 0, ...,
    Sunday" = 6 to weekday = 1 if Monday,..,Friday and 0 else
    """
    date_frame = pd.to_datetime(date_frame, format="%Y-%m-%d %H:%M:%S")
    # From "Monday = 0, ..., Sunday" = 6 to weekday = 1 if Monday,..,Friday and 0 else
    date_frame["weekday"] = date_frame.dt.dayofweek
    return date_frame["weekday"]

def get_month_features(date_frame):
    date_frame = pd.to_datetime(date_frame, format="%Y-%m-%d %H:%M:%S")
    date_frame["month"] = date_frame.dt.month -1
    return date_frame["month"]


def get_daytime_features(date_frame, unit="hours"):
    date_frame = pd.to_datetime(date_frame, format="%Y-%m-%d %H:%M:%S")
    date_frame["time_in_" + unit] = [round(t.hour + t.minute/60.0, 4) for t in date_frame ]
    return date_frame["time_in_" + unit]

def map_to_sinus(df, colname, nr_of_unique_values = None):
    if nr_of_unique_values == None:
        nr_of_unique_values = len(df[colname].unique())
    return np.sin(df[colname]*(2.0 * np.pi/nr_of_unique_values))

def map_to_cosinus(df, colname, nr_of_unique_values = None):
    if nr_of_unique_values == None:
        nr_of_unique_values = len(df[colname].unique())
    return np.cos(df[colname]*(2.0 * np.pi/nr_of_unique_values))

def apply_feature_engineering(tracks, trackspoints):
    tracks_copy = tracks.drop(["linha","rating_weather","rating_bus"],axis=1)
    tracks_copy = tracks_copy.set_index(keys="id")
    tracks_copy["trip_start"], tracks_copy["trip_end"] = get_starting_and_end_time_per_trip(trackspoints)
    # Feature Engineering
    tracks_copy.reset_index(inplace=True)
    tracks_copy["month"] = get_month_features(tracks_copy["trip_start"])
    # Map to unit circle to get the cyclical nature of months
    tracks_copy["month_sinus"] = map_to_sinus(tracks_copy, "month")
    tracks_copy["month_cosinus"] = map_to_cosinus(tracks_copy, "month")

    tracks_copy["weekday"] = get_weekday_features(tracks_copy["trip_start"])
    # Map to unit circle to get the cyclical nature of weekdays
    tracks_copy["weekday_sinus"] = map_to_sinus(tracks_copy, "weekday")
    tracks_copy["weekday_cosinus"] = map_to_cosinus(tracks_copy, "weekday")

    tracks_copy["daytime_start"] = get_daytime_features(tracks_copy["trip_start"])
    tracks_copy["daytime_start_sinus"] = map_to_sinus(tracks_copy,
                                                      "daytime_start",
                                                      nr_of_unique_values=24)
    tracks_copy["daytime_start_cosinus"] = map_to_cosinus(tracks_copy,
                                                          "daytime_start",
                                                          nr_of_unique_values=24)

    tracks_copy["daytime_end"] = get_daytime_features(tracks_copy["trip_end"])
    tracks_copy["daytime_end_sinus"] = map_to_sinus(tracks_copy,
                                                      "daytime_end",
                                                      nr_of_unique_values=24)
    tracks_copy["daytime_end_cosinus"] = map_to_cosinus(tracks_copy,
                                                          "daytime_end",
                                                          nr_of_unique_values=24)

    tracks = tracks_copy.drop(["id", "id_android", "trip_start","trip_end",
                               "weekday","daytime_start","daytime_end", "month"],
                               axis=1)

    #scaled_columns = ["speed","time","distance","weekday_sinus",
    #                  "weekday_cosinus", "daytime_start_sinus",
    #                  "daytime_start_cosinus", "daytime_end_sinus",
    #                  "daytime_end_cosinus"]
    scaled_columns = ["speed","time","distance"]
    tracks_scaled = StandardScaler().fit_transform(tracks[scaled_columns])
    tracks_scaled_df = pd.DataFrame(tracks_scaled, columns=scaled_columns)
    tracks[scaled_columns] = tracks_scaled_df[scaled_columns]

    return tracks
