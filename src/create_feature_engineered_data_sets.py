import os
import pandas as pd
from data.make_data import DataSet
from utils import general_utils
from features.make_features import apply_feature_engineering

"""
    This script applies the identified feature engineering needs and creates
    a new directory data\preprocessed_and_feature_engineered with the preprocessed versions of
    go_track_tracks.csv and go_track_trackspoints.csv (no feature engineering applied yet)
"""
def main():
    parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(parentdir, "data", "preprocessed")
    dataset = DataSet(datadir,file_ending=".csv")
    tracks, trackspoints = dataset.get_data()

    feature_engineered_tracks = apply_feature_engineering(tracks, trackspoints)
    datadir_preprocessed_and_feature_engineered = os.path.join(parentdir, "data", "preprocessed_and_feature_engineered")
    general_utils.setup_directory(datadir_preprocessed_and_feature_engineered)
    feature_engineered_tracks.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,"go_track_tracks.csv"), index = False)
    trackspoints.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,"go_track_trackspoints.csv"), index = False)


if __name__ == '__main__':
    main()
