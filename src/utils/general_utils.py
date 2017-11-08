import pandas as pd
import numpy as np
import datetime as dt
import os

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



def setup_directory(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Created Directory: {}".format(dir_name) )
        except:
            print("Could not create directory: {}".format(dir_name))
