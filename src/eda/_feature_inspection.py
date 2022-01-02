import sys 
import os 

sys.path.insert(0, os.path.abspath(''))

# Plotting
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
sns.set_style("darkgrid")
sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})

# Data structures
import pandas as pd
import numpy as np

# For path referencing
from config.definitions import ROOT_DIR
from scripts.utils import get_data

# Global constants
X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)
features_names = ['refractive_index', 'sodium', 'magnesium', 'aluminium', 'silicone', 'potassium', 'calcium', 'barium', 'iron']

def run_eda():
    _class_distribution()
    _feature_fivenumsummaries()
    _each_class_distribution_per_feature()
    _best_2_pca_comp_scatter()

def _class_distribution():
    train = Counter(y_train) 
    test = Counter(y_test)
    total = train + test

    def normalise(d):
        s = sum(d.values())
        for key in d:
            d[key] /= s
            d[key] *= 100
        return d 

    train = normalise(train)
    test = normalise(test)

    fig, ax = plt.subplots(ncols=2, figsize=(4*2, 3))

    for i, info in enumerate(zip([train, test], ['train_split', 'test_split'])):
        count, name = info
        sns.barplot(x=list(count.keys()), y=list(count.values()), palette="Blues_d", ax = ax[i])
        ax[i].set_title(f"{name.replace('_', ' ').title()} Class Distribution")
        ax[i].set_xlabel('Glass Types')
        if i == 0:
            ax[i].set_ylabel('Relative Frequency (%)')

    fig.savefig('./data/figures/class_distribution.pdf')
    print('Saved class distribution to ./data/figures/class_distribution.pdf')

def _feature_fivenumsummaries():
    data = np.vstack((X_train, X_test))

    summaries = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T

    df = pd.DataFrame(summaries, columns = ['Min','Lower Quartile','Median', 'Upper Quartile', 'Max'], index=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])

    print(df)


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
