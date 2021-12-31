# Plotting
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("darkgrid")
sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})

# Data structures
import pandas as pd
import numpy as np

# For path referencing
from config.definitions import ROOT_DIR
import sys
import os

# Global constants
features_names = ['refractive_index', 'sodium', 'magnesium', 'aluminium', 'silicone', 'potassium', 'calcium', 'barium', 'iron']

def run_eda():

    _each_class_distribution_per_feature()
    _best_2_pca_comp_scatter()


def _each_class_distribution_per_feature():

    train_scaled = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'transformed', 'train', 'X_scaled.csv'), delimiter=',', names=features_names)
    y = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'transformed', 'train', 'y.csv'), delimiter=',', header=None)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 12), sharex=True, sharey=True)
    fig.tight_layout(pad=4.0)

    train_scaled_adj = train_scaled.copy()
    train_scaled_adj['target'] = y

    labels = list(np.unique(y))
    for i in range(len(axs)):
        for j in range(len(axs[i])):

            # Relevant ax
            ax = axs[i][j]
            
            # Relevant feature
            feature = features_names[i*3 + j]
            
            # Plot
            sns.violinplot(data=train_scaled_adj, y=feature, x='target', ax=ax, inner='box')
            
            # Set correspodning labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'{" ".join(feature.split("_")).capitalize()}', size=13)
            ax.set_ylim(-4, 4)

    # Add a big axis, hide frame
    ax = fig.add_subplot(111, frameon=False)

    # Hide tick and tick label of the big axis
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Classes of Glass Fragments", size=15)
    ax.set_ylabel("Standardized values", size=15)
    ax.grid(False)

    # Save figure
    fig.savefig(os.path.join(ROOT_DIR, 'data', 'figures', 'fdist_violin.jpg'))

def _best_2_pca_comp_scatter():

    train_pca = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'transformed', 'train', 'X_pca.csv'), delimiter=',', header=None)
    y = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'transformed', 'train', 'y.csv'), delimiter=',', header=None)

    train_pca_adj = train_pca.copy()
    train_pca_adj['target'] = y

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=train_pca_adj, x=0, y=1, ax=ax, hue='target', palette=sns.color_palette("Paired", n_colors=6))

    ax.set_xlabel('First component', size=15)
    ax.set_ylabel('Second component', size = 15)

    fig.savefig(os.path.join(ROOT_DIR, 'data', 'figures', 'pca_best2_scatter.jpg'))
