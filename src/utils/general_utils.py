import pandas as pd
import numpy as np
import datetime as dt
import os
from scipy.misc import comb
import math
from sklearn.preprocessing import scale

def calculate_total_trip_time(trackspoints):
    """
    Returns a pandas dataframe with the trip time in minutes per track_id
    """
    track_ids = trackspoints["track_id"].unique()
    total_trip_times = list()
    for id in track_ids:
        track = trackspoints[trackspoints["track_id"] == id]
        # make sure that index start at 0
        track = track.reset_index()
        last_entry_index = track.shape[0]-1
        trip_time = track["time"][last_entry_index] - track["time"][0]
        trip_time_in_minutes = np.round(trip_time.seconds/60,decimals=2)
        total_trip_times.append(trip_time_in_minutes)

    trips_per_id = pd.DataFrame({"track_id" : track_ids,
                                 "trip_time_in_minutes" : total_trip_times})
    trips_per_id = trips_per_id.set_index(keys="track_id")
    return trips_per_id

def get_starting_and_end_time_per_trip(trackspoints):
    track_ids = trackspoints["track_id"].unique()
    trip_start_times = list()
    trip_end_times = list()
    for id in track_ids:
        track = trackspoints[trackspoints["track_id"] == id]
        # make sure that index start at 0
        track = track.reset_index()
        trip_start_times.append(track["time"][0])

        last_entry_index = track.shape[0]-1
        trip_end_times.append(track["time"][last_entry_index])

    starting_times_per_trip = pd.DataFrame({"track_id" : track_ids,
                                 "trip_start" : trip_start_times})
    starting_times_per_trip = starting_times_per_trip.set_index(keys="track_id")

    end_times_per_trip = pd.DataFrame({"track_id" : track_ids,
                                 "trip_end" : trip_end_times})
    end_times_per_trip = end_times_per_trip.set_index(keys="track_id")

    return starting_times_per_trip, end_times_per_trip


def get_duplicate_track_recording_candidates(trips_per_id):
    """
    Returns a dictionary with lists of all track_id's which could be duplicates.
    """
    duplicates_mask = trips_per_id.duplicated("trip_time_in_minutes",keep=False)
    duplicates = trips_per_id[duplicates_mask]
    nrofduplicates = duplicates.shape[0]

    duplicate_candidates = dict()
    for row in duplicates.itertuples():
        row_value = round(row[1],2)
        index = row[0]
        if row_value not in duplicate_candidates:
            duplicate_candidates[row_value] = []
            duplicate_candidates[row_value].append(index)
        else:
            duplicate_candidates[row_value].append(index)
    return nrofduplicates, duplicate_candidates

def get_real_duplicate_track_recordings(duplicate_candidates, trackspoints, verbose=False):
    """
    Returns a pandas dataframe with all track_id's which are duplicates_mask
    """
    # Note: Im looping through the dataframe due to convenience and because
    # our data is quite small. Usually looping through the df should be avoided.
    real_duplicates = list()
    for key, track_ids in duplicate_candidates.items():
        if verbose:
            print(key, track_ids)
        for track_id_1 in track_ids:
            track1 = trackspoints[trackspoints["track_id"]==track_id_1].reset_index()
            # if track_id_1 == 26:
            #     print(track1.head())
            for track_id_2 in track_ids:
                if track_id_1 != track_id_2:
                    track2 = trackspoints[trackspoints["track_id"]==track_id_2].reset_index()
                    # if track_id_2 == 27:
                    #     print(track2.head())
                    if (track1.shape[0] == track2.shape[0]) and \
                       (track1["latitude"] == track2["latitude"]).all() and \
                       (track1["longitude"] == track2["longitude"]).all() and \
                       (track1["time"] == track2["time"]).all() and \
                       (track_id_1 not in real_duplicates):
                        if verbose:
                            print(track1.head())
                            print(track2.head())
                            print("append ", track_id_2)
                        real_duplicates.append(track_id_2)




    return real_duplicates


def find_largest_difference(track):
    biggest_diff = dt.datetime.now()
    biggest_diff = biggest_diff-biggest_diff

    start_id = 0
    stop_id = 0
    track_id = track.iloc[0]["track_id"]
    for p in range(1,len(track)):
        #calculate difference
        new_diff = track.iloc[p]["time"] - track.iloc[p-1]["time"]
        if new_diff > biggest_diff:
            biggest_diff = new_diff
            start_id = p-1
            stop_id = p
    largest_diff = [track_id, start_id, stop_id, biggest_diff]
    #print(biggest_diff)
    #print(start_id)
    #print(stop_id)
    return largest_diff

def calculate_ratings_per_time_unit(time, tracks, time_unit="weekday"):
    ratings = pd.DataFrame()
    ratings["time"] = time
    for rating in range(1,4):
        ratings["rating_"+str(rating)] = [1 if i==True else 0 for i in tracks["rating"]==rating]

    ratings = ratings.set_index("time")
    if time_unit == "weekday":
        grouped_rating = ratings.groupby(ratings.index.weekday).sum()
    elif time_unit == "month":
        grouped_rating = ratings.groupby(ratings.index.month).sum()
        grouped_rating.index = grouped_rating.index - 1
    elif time_unit == "daytime":
        grouped_rating = ratings.groupby(ratings.index.hour).sum()

    return grouped_rating

def setup_directory(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Created Directory: {}".format(dir_name) )
        except:
            print("Could not create directory: {}".format(dir_name))


def rescale_gps_data(data, scale_by=1000, standardize=False):
    col_names = list(data.columns.values)
    rating_included = "rating" in col_names
    if rating_included:
        ratings = data["rating"]
        data = data.drop("rating", axis=1)
        col_names.remove("rating")
    for name in col_names:
        if "latitude" or "longitude" in name:
            data[name] = round((data[name]*scale_by) % 1, 7)

    if rating_included and standardize:
        data = pd.DataFrame(scale(data), columns=col_names,
                            index=list(data.index))
    elif standardize:
        data = pd.DataFrame(scale(data), columns=col_names,
                            index=list(data.index))

    if rating_included:
        data["rating"] = ratings
    return data


def calculate_ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) *
             error**k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)


def get_binary_labels(data):
    y = list(data)
    y = [1 if (i==1 or i==2) else 0 for i in y]
    y = np.array(y)
    return y
