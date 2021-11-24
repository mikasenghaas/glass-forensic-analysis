import math
import numpy as np 

from ._node import Node
from ...utils import * 

class DecisionTree:
    """
    Parent Class,not intended for use. Use children classes 
    - DecisionTreeClassifier 
    - DecisionTreeRegressor
    """
    def __init__(self, max_depth=None, algorithm='greedy', max_features='auto', random_state=None):
        # generic data
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        # decision tree metrics
        self.root = None
        self.num_nodes = 0
        self.num_leaf_nodes = 0

        self.criterion = None
        self.max_features = max_features # the number of features to search for in best split (if None: entire feature space)
        self.algorithm = algorithm

        # stopping criterion
        if max_depth == None: self.max_depth = math.inf 
        else: self.max_depth = max_depth

        # set random seed
        np.random.seed(random_state)


    def fit(self, X, y):
        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(X, y)

        self.n, self.p = self.X.shape

        if isinstance(self.max_features, int):
            self.max_features = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features = int(np.ceil(self.p * self.max_features))
        elif self.max_features == 'auto':
            self.max_features = int(np.ceil(np.sqrt(self.p)))
        elif self.max_features == 'log2':
            self.max_features = int(np.ceil(np.log2(self.p)))

        # root node
        self.root = Node(size=self.n, values=np.arange(self.n), depth=1, _type='root')
        self.num_nodes += 1

        self._split(self.root)
        self.fitted = True

    def predict(self, X):
        X = validate_feature_matrix(X)
        n = X.shape[0]

        preds = np.empty(n) 
        for i in range(n):
            curr = self.root
            while not curr.is_leaf():
                if curr.decision(X[i]) == True:
                    curr = curr.right
                else: 
                    curr = curr.left

            preds[i] = curr.prediction

        return preds

    def predict_proba(self, X):
        # undefined for single decision tree
        return np.empty(0)

    def __len__(self):
        return self.num_nodes

    def __str__(self):
        if self.fitted:
            curr = self.root
            q = []
            q.insert(0, curr)
            depth = 0
            s = ''

            while q != []:
                curr = q.pop()
                if curr.depth > depth:
                    depth = curr.depth
                    s += f'\nCurrent Depth: {curr.depth}'
                    s += f"{'='*15}\n"
                s += curr.__str__() + '\n'

                if curr.left != None:
                    q.insert(0, curr.left)
                if curr.right != None:
                    q.insert(0, curr.right)
        else: 
            s = 'Decision Tree is not fitted'

        return s

    def _is_pure(self, node):
        return self.criterion(self.y[node.values]) == 0

    def _check_criterion(self, node):
        depth_not_reached = node.depth <= self.max_depth
        # max_nodes_not_reached = self.num_nodes < self.max_nodes
        #max_leaf_nodes_not_reached = self.num_leaf_nodes < self.max_leaf_nodes 
        can_split = len(np.unique(self.X[node.values], axis=0)) > 1

        return (depth_not_reached and 
                # max_nodes_not_reached and 
                #max_leaf_nodes_not_reached and 
                can_split)

    def _best_split(self, X, y):
        if self.algorithm == 'greedy':
            loss = math.inf 
            best_pair = None 
            feature_space = np.random.choice(list(range(self.p)), size=self.max_features, replace=False)

            for p in feature_space:
                sorted_vals = sorted(list(set(X[:, p])))
                splits = [(sorted_vals[i]+sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]
                for val in splits: 
                    lower_val = X[:, p] < val
                    split1 = y[lower_val]
                    split2 = y[~lower_val]

                    split1_impurity = self.criterion(split1)
                    split2_impurity = self.criterion(split2)

                    weighted_impurity = (split1_impurity * len(split1) + split2_impurity * len(split2)) / self.n

                    if weighted_impurity < loss:
                        loss = weighted_impurity
                        best_pair = (p, val)

            return loss, best_pair

        elif self.algorithm == 'random':
            feature_space = np.random.choice(list(range(self.p)), size=self.max_features, replace=False)

            p = np.random.choice(feature_space)

            sorted_vals = sorted(list(set(X[:, p])))
            while len(sorted_vals) <= 1:
                p = np.random.choice(feature_space)

                sorted_vals = sorted(list(set(X[:, p])))

            splits = [(sorted_vals[i]+sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]

            val = np.random.choice(splits)

            lower_val = X[:, p] < val
            split1 = y[lower_val]
            split2 = y[~lower_val]

            split1_impurity = self.criterion(split1)
            split2_impurity = self.criterion(split2)

            weighted_impurity = (split1_impurity * len(split1) + split2_impurity * len(split2)) / self.n

            return weighted_impurity, (p, val)



    def _split(self, curr):
        # curr is initialised as node with size, indices of values and depth
        # find best split
        X, y = self.X[curr.values], self.y[curr.values] # consider training samples that are in split of current node

        loss, best_pair = self._best_split(X, y) # find best pair to split further

        # assign loss and split criterion
        p, val = best_pair
        curr.loss = loss
        curr.p = p 
        curr.val = val

        # compute new split
        train_decisions = []
        for x in X:
            train_decisions.append(curr.decision(x))
        train_decisions = np.array(train_decisions)

        curr.split = [curr.size - sum(train_decisions), sum(train_decisions)]

        # find new indices in splits
        next_values = [[], []]
        for i in curr.values:
            if curr.decision(self.X[i]) == 0:
                next_values[0].append(i)
            else:
                next_values[1].append(i)

        #next_values = [np.array(next_values[0]), np.array(next_values[1])]

        # initialise new nodes
        curr.left = Node(size=curr.split[0], values=next_values[0], depth=curr.depth+1)
        self.num_nodes += 1

        # split further if not pure or pre-pruning stop criterion not reached
        if not self._is_pure(curr.left) and self._check_criterion(curr.left):
            self._split(curr.left)
        else:
            # otherwise make leaf
            curr.left.type = 'leaf'
            curr.left.prediction = self._evaluate_leaf(curr.left)
            self.num_leaf_nodes += 1

        curr.right = Node(size=curr.split[1], values=next_values[1], depth=curr.depth+1)
        self.num_nodes += 1

        if not self._is_pure(curr.right) and self._check_criterion(curr.right):
            self._split(curr.right)
        else:
            curr.right.type = 'leaf'
            curr.right.prediction = self._evaluate_leaf(curr.right)
            self.num_leaf_nodes += 1
