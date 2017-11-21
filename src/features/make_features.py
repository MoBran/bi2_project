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
    date_frame["month"] = date_frame.dt.month - 1
    return date_frame["month"]


def get_daytime_features(date_frame, unit="hours"):
    date_frame = pd.to_datetime(date_frame, format="%Y-%m-%d %H:%M:%S")
    date_frame["time_in_" +
               unit] = [round(t.hour + t.minute / 60.0, 4) for t in date_frame]
    return date_frame["time_in_" + unit]


def map_to_sinus(df, colname, nr_of_unique_values=None):
    if nr_of_unique_values == None:
        nr_of_unique_values = len(df[colname].unique())
    return np.sin(df[colname] * (2.0 * np.pi / nr_of_unique_values))


def map_to_cosinus(df, colname, nr_of_unique_values=None):
    if nr_of_unique_values == None:
        nr_of_unique_values = len(df[colname].unique())
    return np.cos(df[colname] * (2.0 * np.pi / nr_of_unique_values))


def apply_feature_engineering(tracks, trackspoints, keep_ids=False):
    tracks_copy = tracks.drop(
        ["linha", "rating_weather", "rating_bus"], axis=1)
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
    if keep_ids:
        tracks = tracks_copy.drop(["trip_start", "trip_end",
                                   "weekday", "daytime_start", "daytime_end", "month"],
                                  axis=1)
    else:
        tracks = tracks_copy.drop(["id", "id_android", "trip_start", "trip_end",
                                   "weekday", "daytime_start", "daytime_end", "month"],
                                  axis=1)

    # scaled_columns = ["speed","time","distance","weekday_sinus",
    #                  "weekday_cosinus", "daytime_start_sinus",
    #                  "daytime_start_cosinus", "daytime_end_sinus",
    #                  "daytime_end_cosinus"]
    scaled_columns = ["speed", "time", "distance"]
    tracks_scaled = StandardScaler().fit_transform(tracks[scaled_columns])
    tracks_scaled_df = pd.DataFrame(tracks_scaled, columns=scaled_columns)
    tracks[scaled_columns] = tracks_scaled_df[scaled_columns]

    return tracks

def get_trackspoints_per_time_interval(trackspoints, time_interval):
    result = pd.DataFrame()
    track_ids = trackspoints["track_id"].unique()
    for i, track_id in enumerate(track_ids):
        track = trackspoints[trackspoints["track_id"] == track_id]
        track = track.set_index("time")
        first_entry = track.head(1)
        last_entry = track.tail(1)
        track = track.groupby(pd.TimeGrouper(str(time_interval)+"T",base=24, closed="left")).nth(0)

        track = track.drop(track.head(1).index)
        result = pd.concat([result, first_entry,track,last_entry])

    result = result[np.isfinite(result["track_id"])]
    result["track_id"] =[int(i) for i in result["track_id"]]
    trackspoints_per_time_interval = result.drop("id",axis=1).reset_index()

    return trackspoints_per_time_interval

def transform_trackspoints_to_track_per_time_interval(trackspoints, tracks, time_interval=5, max_trip_duration=None):
    """
    Resample the trackspoints data set to new time intervals in
    minutes and summarize missing values with its nearest neighbour.
    """
    trackspoints["time"] = pd.to_datetime(trackspoints["time"], format="%Y-%m-%d %H:%M:%S")
    trip_start, trip_end = get_starting_and_end_time_per_trip(trackspoints)
    if max_trip_duration == None:
        max_trip_duration = int(max(tracks["time"]) * 60)
    new_column_names = []
    for i in range(0, max_trip_duration+10, time_interval):
        new_column_names.append("latitude_time_" + str(i))
        new_column_names.append("longitude_time_" + str(i))
    new_features = pd.DataFrame(np.zeros((tracks.shape[0],len(new_column_names))),
                                columns=new_column_names)

    new_features["track_id"] = tracks["id"]
    new_features.set_index(keys="track_id", inplace=True)
    trackspoints_per_time_interval = get_trackspoints_per_time_interval(trackspoints,time_interval)

    # Stitching trackspoints_per_time_interval and new_features together
    for i, track_id in enumerate(new_features.index):
        track = trackspoints_per_time_interval[trackspoints_per_time_interval["track_id"]==track_id]
        track.reset_index(inplace=True)
        for idx in list(track.index):
            lat_column = "latitude_time_" + str(idx * time_interval)
            long_column = "longitude_time_" + str(idx * time_interval)

            new_features.loc[track_id,
                             lat_column] = track.loc[idx, "latitude"]
            new_features.loc[track_id,
                             long_column] = track.loc[idx, "longitude"]
    new_features = new_features.loc[:,new_column_names]
    new_features["rating"] = tracks.set_index(keys="id")["rating"]

    return new_features



def transform_trackspoints_to_track_per_quantile(trackspoints, tracks, nr_of_quantiles = 5):
    """
    Split the trackspoints data set to a predefined number of quantiles
    and get each quantile point. This results in a dense representation
    and is independent of the length of a track.
    """
    slice_size = 1.0/nr_of_quantiles
    quantile_range = [round(slice_size*i,2) for i in range(0,nr_of_quantiles+1)]
    result = pd.DataFrame()
    trackspoints_copy = trackspoints
    trackspoints = trackspoints.reset_index()
    for track_id in list(tracks["id"]):
        track = trackspoints[trackspoints["track_id"]==track_id]
        track_quantiles = track.quantile(quantile_range)
        result = pd.concat([result, track_quantiles])

    quantile_indices = [int(i) for i in result["index"]]
    trackspoints_quantiles = trackspoints_copy.iloc[quantile_indices]

    new_column_names = []
    for i in range(0, nr_of_quantiles+1):
        new_column_names.append("latitude_quantile_" + str(i))
        new_column_names.append("longitude_quantile_" + str(i))
    new_features = pd.DataFrame(np.zeros((tracks.shape[0],len(new_column_names))),
                                columns=new_column_names, index=tracks["id"])

    # Stitching quantiles and new_features together
    for i, track_id in enumerate(new_features.index):
        track = trackspoints_quantiles[trackspoints_quantiles["track_id"]==track_id]
        track.reset_index(inplace=True)
        for idx in list(track.index):
            lat_column = "latitude_quantile_" + str(idx)
            long_column = "longitude_quantile_" + str(idx)

            new_features.loc[track_id,
                             lat_column] = track.loc[idx, "latitude"]
            new_features.loc[track_id,
                             long_column] = track.loc[idx, "longitude"]

    new_features = new_features.reset_index().rename(columns={"id":"track_id"}).set_index("track_id")
    return new_features
