
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from itertools import cycle

import seaborn as sns
import matplotlib.patches as mpatches


def extract_class_specific_feature(d, feature, y_label):
    ''' This function creates a dictionary containing a list of values of a specific feature for every instance of the y_label
    d = dataset
    feature = independent variable column name
    y_label = dependent variable column name
    '''
    y_categorical = set(d[y_label])
    feature_dic = {}
    for y_val in y_categorical:
        feature_dic[y_val] = list(d[d[y_label] == y_val][feature])
        #feature_list.append())
    return feature_dic

def create_density_plot_for_categorical_variable(d, feature, y_label):
    ''' creates an overlapping density plot of the feature for every distinct value of y_label
    d = dataset
    feature = independent variable column name
    y_label = dependent variable column name
    '''
    palette = cycle(sns.color_palette())
    y_categorical = set(d[y_label])
    #create a dictionary
    feature_dict = extract_class_specific_feature(d, feature, y_label)
    #create a desnity line for every categorical variable (key)
    handles = []
    data_max = 0
    for key in y_categorical:
        color=next(palette)
        sns.set()
        plot=sns.distplot(feature_dict[key], hist=False, color=color, kde_kws={"shade":True})
        max_y = plot.yaxis.get_data_interval()[1]
        data_max =  max_y if max_y > data_max else data_max
        label = mpatches.Patch(color=color, label=key)
        handles.append(label)
    plt.ylim(0, max_y*1.1)
    plt.rcParams["figure.figsize"] = (6,4)
    plt.title(feature)
    plt.show();

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
    ax.legend(loc="upper center", fontsize=14)

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

def plot_multiclass_ROC_curve(y_test, y_score, title=None):
    """
    Plot a mulitclass ROC curve.
    The code has been adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    micro -  ROC curve by considering each element of the
             label indicator matrix as a binary prediction (micro-averaging).
    macro -  macro-averaging, which gives equal weight to the classification
             of each label.
    """
    if type(y_test) is pd.DataFrame:
        y_test = np.array(y_test)

    n_classes = y_test.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(12,8))
    lw = 2
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):

        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of rating {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))


    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title == None:
        ax.set_title('ROC for multi-class',fontsize=16)
    else:
        ax.set_title(title,fontsize=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16);


def plot_ROC_curve(y_test, y_score, title=None):
    n_classes = y_test.shape[0]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])


    fig, ax = plt.subplots(figsize=(8,4))
    lw = 2
    ax.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title == None:
        ax.set_title('ROC Curve')
    else:
        ax.set_title(title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14);



def plot_validation_curves(train_scores, test_scores, param_range, xlabel = "Parameter"):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5,
         label='training accuracy $\pm \sigma$')
    ax.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    ax.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy $\pm \sigma$')
    ax.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    ax.grid()
    #plt.xscale('log')
    ax.legend(loc='lower right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.2])
    plt.show();
