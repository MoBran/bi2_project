
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_weekdays_polar(df):
    """This code has been adapted from
    http://www.datasciencebytes.com/bytes/2015/12/15/polar-plots-and-shaded-errors-in-matplotlib/
    Plot DataFrame of day-of-week data as a polar plot

    DataFrame should be indexed 0-6 with 0=Monday, 6=Sunday
    """
    mpl.style.use('ggplot')
    # add last row to complete cycle (otherwise plot lines don't connect)
    df = df.append(df.ix[0, :])

    # convert index to radians
    radians = np.linspace(0, 2 * np.pi, num=len(df)-1, endpoint=False)
    df.index = [radians[day] for day in df.index]

    plt.figure(figsize=(6, 6))
    ax = plt.axes(polar=True)
    ax.set_theta_zero_location('N')
    # Set up labels
    ax.set_xticks(radians)
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday',
                        'Thursday', 'Friday', 'Saturday', 'Sunday'],
                         size=14, weight='bold')

    df.plot(ax=ax, lw=4)
    # need to lookup line color for shaded error region
    line_colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    ax.set_title("Traffic Ratings: 3=good, 2=normal, 1=bad \n")
    ax.legend(loc="lower right", fontsize=14)

def plot_daytime_polar(df):
    """This code has been adapted from
    http://www.datasciencebytes.com/bytes/2015/12/15/polar-plots-and-shaded-errors-in-matplotlib/
    Plot DataFrame of daytime data as a polar plot

    DataFrame should be indexed 0-23
    """
    mpl.style.use('ggplot')
    # There are no recordings at 1 o'clock in the morning, thats why I add
    # an empty column at index 1
    df.loc[1] = [0,0,0]
    df = df.sort_index()
    index_copy = df.index
    # add last row to complete cycle (otherwise plot lines don't connect)
    df = df.append(df.ix[0, :])

    # convert index to radians
    radians = np.linspace(0, 2 * np.pi, num=len(df), endpoint=False)
    df.index = [radians[hour] for hour in df.index]

    plt.figure(figsize=(6, 6))
    ax = plt.axes(polar=True)
    ax.set_theta_zero_location('N')
    # Set up labels
    ax.set_xticks(radians)
    ax.set_xticklabels([str(hour) for hour in index_copy],
                         size=14, weight='bold')

    df.plot(ax=ax, lw=4)
    # need to lookup line color for shaded error region
    line_colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    ax.set_title("Traffic Ratings: 3=good, 2=normal, 1=bad \n")
    ax.legend(loc="lower right", fontsize=14)

def plot_months_polar(df):
    """This code has been adapted from
    http://www.datasciencebytes.com/bytes/2015/12/15/polar-plots-and-shaded-errors-in-matplotlib/
    Plot DataFrame of daytime data as a polar plot

    DataFrame should be indexed 0-11, for 0=January,..,11=December
    """
    mpl.style.use('ggplot')
    # add last row to complete cycle (otherwise plot lines don't connect)
    df = df.append(df.ix[0, :])

    # convert index to radians
    radians = np.linspace(0, 2 * np.pi, num=len(df), endpoint=False)
    df.index = [radians[hour] for hour in df.index]

    plt.figure(figsize=(6, 6))
    ax = plt.axes(polar=True)
    ax.set_theta_zero_location('N')
    # Set up labels
    ax.set_xticks(radians)
    ax.set_xticklabels(['January', 'February', 'March', 'April',
                        'May', 'June', 'July', 'August', 'September',
                        'October', 'November', "December"],
                         size=14, weight='bold')

    df.plot(ax=ax, lw=4)
    # need to lookup line color for shaded error region
    line_colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    ax.set_title("Traffic Ratings: 3=good, 2=normal, 1=bad \n")
    ax.legend(loc="lower center", fontsize=14)
