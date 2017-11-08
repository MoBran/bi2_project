# VU Business Intelligence II Project - University of Vienna

This project performs an analysis and predictions of the GPS data set from the BI 2 course.

Abstract: The dataset has been feed by Android app called Go!Track.
Source:
M. O. Cruz, H. T. Macedo, R. Barreto

View the [data exploration](src/DataExploration.ipynb)

View the [model building](src/model_building.ipynb) (**still work in progress**)

You can view the source code [here](src)

## Requirements for running the jupyter notebooks on windows:

**It is assumed that you have the two csv files go_track_tracks.csv and go_track_trackspoints.csv
in the [data](data) dir.**

1. Download and install [anaconda](https://www.anaconda.com/download/)
2. Create a conda environment by entering:
   ```
   conda create --name Name python=3 numpy=1.13 pandas=0.20 scikit-learn=0.19 matplotlib=2.0 seaborn 0.8
   ```
3. Activate the environment by entering:
   ```
   activate Name
   ```
4. Create preprocessed data by entering:
   ```
   bi2_project/src> python create_data_sets.py
   ```
5. Now you can execute the jupyter notebook [DataExploration.ipynb](src/DataExploration.ipynb) and [model_building.ipynb](src/model_building.ipynb)
