import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from ..utils import COLORS

def plot_1d_decision_regions(x, y, model, meshsize=0.01, title='unnamed', ax=None):

    """
    Plots 1d decision regions for given feature and model.
    In addition, it adds on top training datapoints.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        One dimensional array with values of the given feature.
    y : :class:`numpy.ndarray`
        One dimensional array with labels of given classes from training dataset.
    model : python object
        Trained model.
    meshsize : float, optional
        Space between points wihthin the meshgrid.
    title : str, optional
        Title of the plot.
    ax : :class:`matplotlib.axes.Axes`, optional
        If you want to plot it to the prespecified axes.
    
    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The plotted 1d decision region.
    
    Raises
    ------
    AssertionError
        If the feature space is not one dimensional.
    """

    # Setup
    assert len(x.shape) == 1, 'Feature Space must be 1-dimensional'
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n, p  = len(y), 1
    k = len(np.unique(y))

    # Define styling properties
    colormap = np.array(COLORS[:k])
    labels = np.array([f'Class {i+1}' for i in range(k)])

    # Meshgrid plotting for decision boundary
    PAD = 0.1 # relative padding
    x_min, x_max = x.min()*(1-PAD), x.max()*(1+PAD)
    x_range = np.arange(x_min, x_max, meshsize)
    y_min, y_max = -5*(1-PAD), 5*(1+PAD)
    y_range = np.arange(y_min, y_max, meshsize)

    xx0, xx1 = np.meshgrid(x_range, y_range)
    xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))

    # Get predictions for each datapoint in meshgrid
    pred_y = [model.intcode[trg] for trg in model.predict(xx[:,0])]

    # Plot the regions
    ax.scatter(xx[:, 0], xx[:, 1], c=colormap[pred_y], alpha=0.3)

    # Plot the training points
    for k, color, label in zip(np.unique(y), colormap, labels):
        ax.scatter(x[y==k], np.ones(sum(y==k)), c=color, s=20, edgecolor='black', linewidth=1, label=label)

    # Plot labels
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$X_1$')
    ax.legend(loc='best')

    return fig

def plot_2d_decision_regions(x, y, model, meshsize=100, marker='s', colors=None, show_probs=False, title='unnamed', ax=None):
    """
    Plots 2d decision regions for given 2 features and model.
    In addition, it adds on top training datapoints.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        2 dimensional array with values of the given features.
    y : :class:`numpy.ndarray`
        One dimensional array with labels of given classes from training dataset.
    model : python object
        Trained model.
    meshsize : float, optional
        Space between points wihthin the meshgrid.
    marker : str, optional
        Type of the marker to represent datapoints in the plot.
    colors : list, optional
        List with colors for each feature.
    show_probs : bool
        Intensity of the regions color is based on the probability of the predicted class.
    title : str, optional
        Title of the plot.
    ax : :class:`matplotlib.axes.Axes`, optional
        If you want to plot it to the prespecified axes.
    
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The plotted 2d decision region.
    
    Raises
    ------
    AssertionError
        If the feature space is not two dimensional.
    """

    # Global properties of data
    assert len(x.shape) == 2, 'Feature Space must be 2-dimensional'
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n, p = x.shape
    k = len(np.unique(y))

    # Define styling properties
    if colors == None:
        colormap = np.array(COLORS[:k])
    else:
        assert k == len(colors), 'enter correct number of colors'
        colormap = np.array(colors)

    if hasattr(model, 'classes'):
        labels = np.array([f'Class: {str(c).title()}' for c in model.classes()])
    else:
        labels = np.array([i for i in range(k)])

    # Prepare data
    x0 = x[:, 0]
    x1 = x[:, 1]

    y = np.array([model.intcode[trg] for trg in y])

    # Meshgrid plotting for decision boundary
    PAD = 0 # relative padding
    x0_min, x0_max = x0.min()*(1-PAD), x0.max()*(1+PAD)
    x0_range = np.linspace(x0_min, x0_max, meshsize)
    x1_min, x1_max = x1.min()*(1-PAD), x1.max()*(1+PAD)
    x1_range = np.linspace(x1_min, x1_max, meshsize)

    xx0, xx1 = np.meshgrid(x0_range, x1_range)
    xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))

    # Get predictions for each datapoint in meshgrid 
    pred_y = np.array([model.intcode[trg] for trg in model.predict(xx)])

    if hasattr(model, 'predict_proba') and show_probs==True:
        proba_y = np.array([max(proba) for proba in model.predict_proba(xx)])
        ax.scatter(xx[:, 0], xx[:, 1], marker=marker, c=colormap[pred_y], s=10**2, alpha=proba_y**2)
    else: 
        ax.scatter(xx[:, 0], xx[:, 1], marker=marker, c=colormap[pred_y], s=10**2, alpha=1)

    # Scatter training points
    for k, color, label in zip(np.unique(y), colormap, labels):
        ax.scatter(x0[y==k], x1[y==k], c=color, s=20, edgecolor='black', linewidth=.5, label=label)

    # Labels
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.legend(loc='best')

    training_acc = AnchoredText(f'Training Accuracy: {round(model.score() * 100, 2)}%', loc=4)
    ax.add_artist(training_acc)

    return ax
