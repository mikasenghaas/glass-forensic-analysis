import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions

from scripts.models import DecisionTreeClassifier
from scripts.metrics import accuracy_score

from scripts.plotting import plot_2d_decision_regions

from sklearn.datasets import load_iris
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    # iris data 
    X, y = load_iris(return_X_y=True)
    X=X[:, :2]

    # make dataset unique to ensure 100% training accuracy for max_depth 
    uniq_idx = np.unique(X[:, :2], return_index=True, axis=0)[1]
    X = X[uniq_idx, :2]
    y = y[uniq_idx]
    #y = y[y!=2]

    """
    # scale features for gradient descent to work properly
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    """

    # train test split to evaluate out-of-bag-performance
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialise and train model
    clf = DecisionTreeClassifier(max_depth=None, max_features=2)
    clf.fit(X_train, y_train)

    # get predictions for training and test split
    train_preds = clf.predict(X_train)
    print(train_preds)
    test_preds = clf.predict(X_test)

    # evaluate performance
    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}')

    fig = plot_2d_decision_regions(X_train, y_train, clf, meshsize=0.03)
    plt.show()


if __name__ == '__main__':
    main()
