import os
import pandas as pd
from data.make_data import DataSet
from utils import general_utils
from features.make_features import apply_feature_engineering
from features.make_features import transform_trackspoints_to_track_per_quantile
from features.make_features import transform_trackspoints_to_track_per_time_interval




"""
    This script applies the identified feature engineering needs and creates
    a new directory data\preprocessed_and_feature_engineered with the preprocessed versions of
    go_track_tracks.csv and go_track_trackspoints.csv (no feature engineering applied yet)
"""
def main():
    ##############
    #
    # parameters
    #
    ##############
    # time_interval and max_trip_duration in minutes
    time_interval = 10
    max_trip_duration = 60
    # for the other table you can specify the number of quantiles
    nr_of_quantiles = 4

    parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(parentdir, "data", "preprocessed")
    dataset = DataSet(datadir,file_ending=".csv")
    tracks, trackspoints = dataset.get_data()

    feature_engineered_tracks = apply_feature_engineering(tracks, trackspoints, keep_ids=True)
    trackspoints_per_time_interval = transform_trackspoints_to_track_per_time_interval(trackspoints,
                                                                                       feature_engineered_tracks,
                                                                                       time_interval=time_interval,
                                                                                       max_trip_duration=max_trip_duration)
    trackspoints_per_quantile = transform_trackspoints_to_track_per_quantile(trackspoints,feature_engineered_tracks,
                                                                              nr_of_quantiles=nr_of_quantiles)

    feature_engineered_tracks = feature_engineered_tracks.drop(["id", "id_android"],axis=1)
    datadir_preprocessed_and_feature_engineered = os.path.join(parentdir, "data", "preprocessed_and_feature_engineered")
    general_utils.setup_directory(datadir_preprocessed_and_feature_engineered)
    feature_engineered_tracks.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,"go_track_tracks.csv"), index = False)
    trackspoints.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,"go_track_trackspoints.csv"), index = False)

    datadir_preprocessed_and_feature_engineered = os.path.join(datadir_preprocessed_and_feature_engineered,"trackspoints")
    general_utils.setup_directory(datadir_preprocessed_and_feature_engineered)
    trackspoints_per_time_interval.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,
                                                       "trackspoints_per_time_interval_{}_and_max_time_{}.csv".format(str(time_interval),str(max_trip_duration))), index = True)
    trackspoints_per_quantile.to_csv(os.path.join(datadir_preprocessed_and_feature_engineered,
                                                  "trackspoints_per_quantile_{}.csv".format(str(nr_of_quantiles))), index = True)


if __name__ == '__main__':
    main()
