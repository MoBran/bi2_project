import os
import pandas as pd
from data.make_data import DataSet
from utils import general_utils
"""
    This script applies the identified preprocessing issues and creates
    a new directory data\preprocessed with the preprocessed versions of
    go_track_tracks.csv and go_track_trackspoints.csv
"""
def main():
    parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(parentdir, "data")
    dataset = DataSet(datadir,file_ending=".csv")
    tracks, trackspoints = dataset.get_data()

    tracks, trackspoints = dataset.preprocess_gps_data()
    datadir_preprocessed = os.path.join(datadir,"preprocessed")
    general_utils.setup_directory(datadir_preprocessed)
    tracks.to_csv(os.path.join(datadir_preprocessed,"go_track_tracks.csv"), index = False)
    trackspoints.to_csv(os.path.join(datadir_preprocessed,"go_track_trackspoints.csv"), index = False)


if __name__ == '__main__':
    main()
