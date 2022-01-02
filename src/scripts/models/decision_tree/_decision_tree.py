import math
from collections import Counter
import numpy as np

from ._node import Node
from ...utils import validate_feature_matrix, validate_target_vector, check_consistent_length
from ...metrics import gini, entropy


class DecisionTree:

    """DecisionTree data structure which can be utilized for Classification or Regression.

    This class serves as a core data structure for implementing decision tree algorithm.
    Its main functionality is to build the tree and then allow to traverse it in order
    to make predictions.

    Parameters  
    ----------
    criterion : str, optional
        How to calculate impurity of nodes.
    
    algorithm : str, optional
        How to approach building process of the tree.
    
    max_depth : int, optional
        Maximum depth of the tree.

    max_features : int or float or str, optional
        How many features to consider during each split of the node.
    
    min_samples_split : int, optional
        Minimum samples required to be present within the node in order
        to be able to further split it.
    
    random_state : int, optional
        If using `random` alforithm, it is useful to specify this parameter
        in order to ensure reproducibility of results.

    Attributes
    ----------
    root : :class:`Node`
        Root node of this tree.
    
    num_nodes : int
        Number of nodes that this tree has.
    
    num_leaf_nodes : int
        Number of leaf nodes that this tree has.
    """

    def __init__(
            self, 
            algorithm='greedy',
            criterion='gini', 
            max_depth=None, 
            max_features='auto',
            min_samples_split=2,
            random_state=None):

        # User defined attributes
        self.algorithm = algorithm
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth if max_depth else math.inf
        np.random.seed(random_state)

        # Run additional setup based on inputed parameters
        self._run_init_setup()

        # Information about this tree
        self.root = None
        self.num_nodes = 0
        self.num_leaf_nodes = 0

    def _run_init_setup(self):

        """Run neccessary proccesses for initialization of the class.

        Notes
        -----
        The neccessary processes are:

        - Assignment of relevant criterion function
        """

        # Assignment of relevant criterion function
        if self.criterion == 'gini':
            self.criterion = gini
        elif self.criterion == 'entropy':
            self.criterion = entropy
        else: 
            raise Exception('Cannot find this criterion')

    def _run_training_setup(self, X, y):

        """Run preliminary setup before the start of training.

        Parameters
        ----------
        X : 2d-array
            Training dataset
        y : 1d-array
            Target values.

        Notes
        -----
        The following things are done:

        - Validation of input tensors X and y - X must be 2 dimensional. If it is 1 dimensional, transformation to 2D is attempted. y must be 1 dimensional.

        - X and y must have identical first dimension

        - Parse number of training records `n`, size of feature space `p` and # of `unique` classes `k`

        - Map provided unique classes to internal normalized format, i.e., 0, 1, 2, ...

        - Determination of max features variable

        """

        self.X = validate_feature_matrix(X)
        self.y = validate_target_vector(y)
        check_consistent_length(X, y)

        self.n, self.p = self.X.shape
        unique_classes = np.unique(self.y)
        self.k = len(unique_classes)

        self.label = {k: unique_classes[k] for k in range(self.k)}
        self.intcode = {unique_classes[k]:k for k in range(self.k)}

        if isinstance(self.max_features, int):
            self.max_features = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features = int(np.ceil(self.p * self.max_features))
        elif self.max_features == 'auto':
            self.max_features = int(np.ceil(np.sqrt(self.p)))
        elif self.max_features == 'log2':
            self.max_features = int(np.ceil(np.log2(self.p)))
        elif self.max_features == None or self.max_features == 'max':
            self.max_features = self.p
        else:
            Exception('Dont know this max_feature option.')

    def fit(self, X, y):

        """Build decision tree.

        Parameters
        ----------
        X : 2d-array
            Training dataset
        y : 1d-array
            Target values.
        """

        # Run training setup
        self._run_training_setup(X, y)

        # Initiliaze a root node
        self.root = Node(size=self.n, values=np.arange(self.n), depth=1, _type='root')
        self.num_nodes += 1

        # Build the tree
        self._split(self.root)
        self.fitted = True

    def predict_proba(self, X):

        """Return 2d array with probabilities for given classes.

        Traverse the tree until you reach a leaf node. Then,
        return the probabilites associated with all classes.

        Parameters
        ----------
        X : 2d array
            Samples based on which to predict corresponding class probabilities.

        Returns
        -------
        2d array
            n x k array where n is number of provided samples and k is number of classes.
        """

        X = validate_feature_matrix(X)
        n = X.shape[0]

        probs = [] 
        for i in range(n):
            curr = self.root
            while not curr.is_leaf():
                if curr.left and curr.right:
                    if curr.decision(X[i]):
                        curr = curr.left
                    else: 
                        curr = curr.right
                else:
                    curr = curr.left or curr.right

            probs.append(curr.predict_proba)

        return np.array(probs)

    def predict(self, X):

        """Predict classes for given dataset X.

        Parameters
        ----------
        X : 2d-array
            Dataset based on which you want to make a prediction.
        
        Returns
        -------
        1d-array
            Returns array with predictions.
        """
        return np.vectorize(self.label.get)(np.argmax(self.predict_proba(X), axis=1))

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
                    s += f'\nCurrent Depth: {curr.depth}\n'
                    s += f"{'='*15}\n"
                s += curr.__str__() + '\n'

                if curr.left != None:
                    q.insert(0, curr.left)
                if curr.right != None:
                    q.insert(0, curr.right)
        else: 
            s = 'Decision Tree is not fitted'

        return s


    def _best_split(self, X, y):

        """Find best split from given feature space.

        This methods finds best split from given feature space
        using given criterion (e.g. gini).

        Parameters
        ----------
        X : 2d-array
            Subset of the original dataset (self.X)
        y : 1d-array
            Subset of the original targets (self.y)
        
        Returns
        -------
        loss : float
            Measure of how good the split of this node is.
        p : int
            Index of the feature according to which the split is supposed to be done.
        val : float or int
            Value according which to do the split.
        """

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

            return loss, best_pair[0], best_pair[1]
        
        else:
            raise Exception('You have specified algorithm which is not implemented.')
    
    def _get_child_info(self, node, X):

        """Get info about given node's children.

        By info, it is precisely meant size of both children
        and relevant indices.

        Parameters
        ----------
        node : :class:`Node`
            Parent node.
        X : 2d array
            Subset of the original training dataset (self.X)

        Returns
        -------
        sizes : list
            List with size of both children.
        
        left_indices : 1d array
            Indices of samples in the left node.

        right_indices : 1d array
            Indices of samples in the right node.
        """

        # Compute new split
        train_decisions = []
        for x in X:
            train_decisions.append(node.decision(x))
        train_decisions = np.array(train_decisions)

        sizes = node.split = [sum(train_decisions), node.size - sum(train_decisions)]

        # Find new indices in splits
        next_values = [[], []]
        for i in node.values:
            if node.decision(self.X[i]):
                next_values[0].append(i)
            else:
                next_values[1].append(i)
        left_indices, right_indices = next_values

        return sizes, np.array(left_indices), np.array(right_indices)

    
    def _is_pure(self, node):

        """Return if given node is pure or not.

        Pure means that the given node only contains
        samples with the same class.

        Parameters
        ----------
        node : :class:`Node`
            Leaf node which you want to evaluate.

        Returns
        -------
        bool
            Is given node pure or not.
        """

        self.criterion(self.y[node.values]) == 0

    def _check_criterion(self, node):

        """Check criteria neccessary to split the given node.

        Parameters
        ----------
        node : :class:`Node`
            Leaf node which you want to evaluate.

        Returns
        -------
        bool
            Can you split the given node.
        """

        # Criteria
        not_pure = not self._is_pure(node)
        depth_not_reached = node.depth <= self.max_depth
        min_samples_not_reached = len(node.values) >= self.min_samples_split
        can_split = len(np.unique(self.X[node.values], axis=0)) > 1

        # Result
        res = depth_not_reached and min_samples_not_reached and can_split and not_pure

        return res

    def _evaluate_leaf(self, node):

        """Return predicted class and probabilities of all classes.

        Parameters
        ----------
        node : :class:`Node`
            Leaf node which you want to evaluate.
        
        Returns
        -------
        predict : int
            Predicted class.
        predict_proba : list
            Probabilities of all classes
        """

        labels = self.y[node.values]
        counter = Counter(labels)
        predict = counter.most_common()[0][0] # most_frequent class

        predict_proba = [0 for _ in range(self.k)]
        for pred, c in counter.items():
            predict_proba[self.intcode[pred]] = c / sum(counter.values())

        return predict, predict_proba

    def _split(self, curr):

        """Split provided node.

        High level overview of algorithm (see further explanation in Notes):

        #. Compute optimal split for the given :class:`Node`.
        #. Compute info about two child nodes which is needed to initiliaze them
        #. Initialize child nodes and for each

          #. IF ALLOWED split it further
          #. ELSE Make it leaf node
        
        Repeat the same process `recursively`.

        Parameters
        ----------
        curr : :class:`Node`
            Current node which should be splitted.
        
        Notes
        -----

        **STEP 1.** In order to find optimal split of the node, the following
        pseudo algorithm is used:

        #. Define proper feature space, i.e. features to be considered for the split. This is determined based on the parameter ``self.max_features``.

        #. For each feature within the proper feature space:

           #. Sort its values and make sure they are all unique
           #. For each pair of neighboring values, compute the split
           #. For this split compute the **total score** - weighted average of given ``criterion``
           #. Return the best split based on the total score
        
        #. From the best splits for each feature, choose the overall best split and return it

        **STEP 2.** Following steps are done:

        - Get boolean array which represents the split of data - True = left node, False = right node
        - Get indices of samples which should belong to left and right child nodes respectively

        
        **STEP 3.** There are two possible scenario for the node in question:
        
        #. split it further
        #. make it leaf node

        Split it further if the node:

        - is NOT pure (i.e. does not contain just one type of class)
        - its depth is not equal to maximum depth specified
        - Contains more samples than the specified minimum
        - Has more than 1 unique sample

        Otherwise, turn it into a leaf node.

        """

        # -- STEP 1 ----------------------------------------------------------------
        X, y = self.X[curr.values], self.y[curr.values]
        curr.loss, curr.p, curr.val = self._best_split(X, y)

        # -- STEP 2 ----------------------------------------------------------------
        sizes, left_indices, right_indices = self._get_child_info(curr, X)

        # -- STEP 3 ----------------------------------------------------------------
        children = []
        for node_size, samples_indices in zip(sizes, [left_indices, right_indices]):
            
            # Avoid empty nodes
            if node_size == 0:
                children.append(None)
                continue

            # Initialize new node
            child = Node(size=node_size, values=samples_indices, depth=curr.depth+1)
            children.append(child)
            self.num_nodes += 1

            # Further split or leaf
            if self._check_criterion(child):
                self._split(child)
            else:
                child.type = 'leaf'
                child.predict, child.predict_proba = self._evaluate_leaf(child)
                self.num_leaf_nodes += 1
        
        curr.left, curr.right = children
