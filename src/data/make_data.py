import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime as dt
# Self written packages
from utils import general_utils


class DataSet:
    """
    Class for handling the loading and preprocessing of the data
    """

    def __init__(self, data_dir_name, file_ending=".csv"):
        self.data_dir_name = data_dir_name
        self.file_ending = file_ending
        self.data_sets = None

    def _get_file_names(self, dir_name):
        dir_content = os.listdir(dir_name)
        csv_file_names = list()
        for file in dir_content:
            if file.endswith(self.file_ending):
                csv_file_names.append(file)
        return csv_file_names


    def get_data(self,ReadIndexColumn=False):
        """ Reads data with a specified file_ending
        Args:
          data_dir_name: Name of directory in the immutable data folder from which
                         the specified files should be read
        Returns:
          Returns a pandas DataFrame of the file in the directory.
          If there are multiple files, a list of DataFrames is
          returned.
        """
        file_names = self._get_file_names(self.data_dir_name)
        if len(file_names) == 1:
            file_name = file_names[0]
            data_file_dir = os.path.join(self.data_dir_name, file_name)
            data = pd.read_csv(data_file_dir, index_col=ReadIndexColumn)
        elif len(file_names) > 1:
            data = list()
            for file_name in file_names:
                data_file_dir = os.path.join(self.data_dir_name, file_name)
                print("Read ", data_file_dir)
                data.append(pd.read_csv(data_file_dir, index_col=ReadIndexColumn))
        else:
            data = pd.DataFrame()
        self.data_sets = data
        return data

    def preprocess_data(self, data, scale_data=True):
        """
        Preprocessing steps for data
        """
        print("preprocess_data not implemented")
        return data

    def _preprocess_trackspoint_data(self, trackspoint_data):
        trackspoint_data["time"] = pd.to_datetime(trackspoint_data["time"])
        return trackspoint_data

    def _preprocess_tracks_data(self, tracks_data):
        tracks_data["linha"] = tracks_data["linha"].fillna("no_answer")
        return tracks_data

    def preprocess_gps_data(self):
        tracks = self._preprocess_tracks_data(self.data_sets[0]);
        trackspoints = self._preprocess_trackspoint_data(self.data_sets[1])
        # Calculate new trip times
        trips_per_track_id = general_utils.calculate_total_trip_time(trackspoints)
        tracks["time"] = trips_per_track_id["trip_time_in_minutes"].reset_index(drop=True)
        # Remove duplicate values
        from utils.general_utils import get_duplicate_track_recording_candidates, \
                                        get_real_duplicate_track_recordings
        _, duplicate_candidates=get_duplicate_track_recording_candidates(trips_per_track_id)
        duplicates = get_real_duplicate_track_recordings(duplicate_candidates,
                                                         trackspoints, verbose=False)

        # Find tracks with large time lags

        from utils.general_utils import find_largest_difference
        diffs = pd.DataFrame(columns=["track_id","start_id","stop_id","diff"])

        max_id = trackspoints.loc[trackspoints["track_id"].idxmax()]["track_id"]

        for id in range(1,max_id):
            #print(id)
            df_helper = trackspoints[trackspoints["track_id"] == id]
            if len(df_helper) != 0:
                diffs.loc[id] = find_largest_difference(df_helper)

        to_be_dropped_large_lag_ids = list(diffs[diffs["diff"]>dt.timedelta(minutes=5)].index)

        # Id's to be removed duplicates, 0 values and large lags
        track_ids_to_be_removed = duplicates + duplicate_candidates[0.0] + to_be_dropped_large_lag_ids
        # Make sure that there are only unique track ids
        track_ids_to_be_removed = list(set(track_ids_to_be_removed))

        tracks = tracks.set_index(keys="id").drop(track_ids_to_be_removed).reset_index()


        # Remove the above track_ids from trackspoints data as well
        duplicate_ids = []
        for track_id in track_ids_to_be_removed:
            duplicate_copy = trackspoints[trackspoints["track_id"] == track_id]["id"]
            duplicate_ids.append(duplicate_copy.values.tolist())
        to_be_dropped_duplicates = []
        for i in duplicate_ids:
            to_be_dropped_duplicates = to_be_dropped_duplicates + i
        trackspoints = trackspoints.set_index(keys="id").drop(to_be_dropped_duplicates).reset_index()

        # Create binary variable for car_or_bus in tracks data
        dummies = pd.get_dummies(tracks['car_or_bus']).rename(columns=lambda x: 'car' + str(x))
        tracks = pd.concat([tracks, dummies], axis=1)
        tracks = tracks.drop(['car_or_bus',"car2"], axis=1)
        tracks = tracks.rename(columns={'car1': 'car_or_bus'})

        return tracks, trackspoints
