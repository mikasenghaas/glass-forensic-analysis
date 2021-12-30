class Node:

    """ 
    Core building block for a Decision Tree.

    The main purpose of a Node class is to hold information about decisions of the
    given decision tree. See what is meant by information in the below description
    of attributes.

    Attributes
    ----------
    size : int, optional
        Number of samples which are part of this node.
    
    values : iterable, optional
        Row indices of the samples which are part of this node.
    
    depth : int, optional
        Depth of this node in decision tree.
    
    _type : str, optional
        What type of node this node is - root, internal, leaf.
    
    p : int
        Index of a feature within the given training dataset. Split of a node is
        made according to this feature.
    
    val : float or int
        Split the incoming dataset according to the ``p-th feature`` and its
        ``value (val)``.
    
    loss : float or int
        Measure of how good the split of this node is. For example, when using
        gini impurity, this would be a weighted average of gini impurities of its
        child nodes.
    
    split : list
        Amount of samples in left and right child of this node.
    
    left : :class:`Node`
        Left child of this node.

    right : :class:`Node`
        Right child of this node.

    predict : int or str
        Predicted class based on the frequency of this class within this node. (Most common)
    
    predict_proba : list
        Probabilites of the classes in question.

    """

    def __init__(self, size=None, values=None, depth=None, _type='internal'):

        # User inputed attributes
        self.size = size
        self.values = values
        self.depth = depth
        self.type = _type
        
        # Information about the split of this node
        self.p = None
        self.val = None
        self.loss = None
        self.split = [None, None]
        self.left = None
        self.right = None

        # Attributes only relevant for leaf nodes
        self.predict = None
        self.predict_proba = None
    
    def decision(self, X):

        """
        Decide whether the given sample(s)' p-th feature value
        `is smaller` than given threshold.

        Parameters
        ----------
        x : nd-array
            Samples present within this node. (Actual values)
        
        Returns
        -------
        1d-array
            Boolean array.
        """

        if len(X.shape) == 1:
            return X[self.p] < self.val
        elif len(X.shape) == 2:
            return X[:, self.p] < self.val
        else:
            raise Exception('Node can make only a decision on 1 or 2 dimensional arrays.')

    def is_leaf(self):
        
        return self.type == 'leaf'

    def __str__(self):

        if self.type == 'root':
            return 'Root Node at Depth 1 '\
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'X[{self.p}] < {self.val}; Splitting {self.size} values in ' \
                   f'[False=={self.split[0]}, True=={self.split[1]}]'

        elif self.type == 'internal':
            return f'Internal Node at Depth {self.depth} ' \
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'X[{self.p}] < {self.val}; Splitting {self.size} values in '\
                   f'[Left=={self.split[0]}, Right=={self.split[1]}]'

        elif self.type == 'leaf':
            return f'Leaf Node at Depth {self.depth} '\
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'Samples in Leaf: {len(self.values)} ' \
                   f'Prediction: {self.predict}'
