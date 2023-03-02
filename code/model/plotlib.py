import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
from pathlib import Path

def load_fonts(path_to_dir):
    """ Recursively load OTF and TTF font files from a directory
        to matplotlib.
    """
    for path in Path(path_to_dir).rglob('*.otf'):
        matplotlib.font_manager.fontManager.addfont(str(path))
    for path in Path(path_to_dir).rglob('*.ttf'):
        matplotlib.font_manager.fontManager.addfont(str(path))


def class_boxplots(df, cols, by, name):
    """ Make boxplots of y classes for each X column.
        
        xcols: list of X features to plot the classes.
        by: columns where the y classes are specified.
        name: path to save the plot.
    """
    n = len(cols)
    if n == 2:
        r = 1
        c = 2
    elif n == 4:
        r = 2
        c = 2
    elif n == 6:
        r = 2
        c = 3
    else:
        c = 3
        r = n // c
    height = 1.2
    if r > 1:
        height = r * 1.0
    fig = plt.figure(figsize = (2*c, height))
    for i in range(n):
        col = cols[i]
        ax = fig.add_subplot(r, c, i+1)
        sns.boxplot(x = by, y = col, data = df, linewidth=0.6)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(name)
    print("Save OK:", name, "Height:", height)
    plt.show()


def corrplot(df, method, figsize = (7, 7), output = None):
    """ Plot a correlation matrix using seaborn """
    corr = df.corr(method=method, numeric_only=True)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 0, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0, vmax=1, vmin=-1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title("%s correlation" %method.capitalize())
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=600)
        plt.close()
        print("Save OK:", output)
    else:
        plt.show()


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          other_stat='',
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization. From https://github.com/DTrimarchi10/confusion_matrix

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    other_stat:    Additional information to add to the summary statistics text.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    stats_text += other_stat

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)