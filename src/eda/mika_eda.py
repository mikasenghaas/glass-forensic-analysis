import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

from scripts.utils import get_data

sns.set(style="darkgrid")

X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)

CLASS_DIST = False
FEATURE_EDA = True


# class distribution plot
if CLASS_DIST:
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
#total = normalise(total)

    fig, ax = plt.subplots(ncols=2, figsize=(4*2, 3))

    for i, info in enumerate(zip([train, test], ['train_split', 'test_split'])):
        count, name = info
        sns.barplot(x=list(count.keys()), y=list(count.values()), palette="Blues_d", ax = ax[i])
        #sns.barplot(x=list(count.keys()), y=list(count.values()), color="blue", saturation=list(count.values()), ax = ax[i])
        ax[i].set_title(f"{name.replace('_', ' ').title()} Class Distribution")
        ax[i].set_xlabel('Glass Types')
        if i == 0:
            ax[i].set_ylabel('Relative Frequency (%)')

    fig.savefig('./data/figures/class_distribution.pdf')
    print('Saved')
#plt.show()


if FEATURE_EDA:
    data = np.vstack((X_train, X_test))

    summaries = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T

    df = pd.DataFrame(summaries, columns = ['Min','Lower Quartile','Median', 'Upper Quartile', 'Max'], index=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
    print(df)
