import os
import numpy as np
import pandas as pd
from sklearn import preprocessing



class DataSet:
    """
    Class for handling the loading and preprocessing of the data
    """

    def __init__(self, data_dir_name, file_ending=".csv"):
        self.data_dir_name = data_dir_name
        self.file_ending = file_ending

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
          If there are multiple files, list of DataFrames is
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
        return data


    def preprocess_data(self, data, scale_data=True):
        """
        Preprocessing steps for data
        """
        if scale_data:
            data_scaled = preprocessing.scale(data)
        return data_scaled

    def preprocess_trackspoint_data(self, trackspoint_data):
        trackspoint_data["time"] = pd.to_datetime(trackspoint_data["time"])
        return trackspoint_data

    def preprocess_tracks_data(self, tracks_data):
        tracks_data["linha"] = tracks_data["linha"].fillna("no_answer")
        return tracks_data
