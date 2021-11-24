class Node:
    def __init__(self, size=None, values=None, depth=None, _type='internal'):
        self.size = size # number of samples to be split in the node
        self.depth = depth # depth of node in decision tree
        self.values = values # indices of the sample to be split in the node

        self.p = None # feature id 
        self.val = None # value to split at (default decision True, if data point lower than val -> in tree vis lower right branch)
        self.loss = None
        self.decision = lambda x: x[self.p] < self.val  # lambda function 

        self.split = [None, None] # amount of samples in split1 and split2values
        self.left = None # if decision evaluates False
        self.right = None # if decision evaluates True

        # leaf node
        self.type = _type
        self.prediction = None

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
                   f'[False=={self.split[0]}, True=={self.split[1]}]'

        elif self.type == 'leaf':
            return f'Leaf Node at Depth {self.depth} '\
                   f'(Loss: {round(self.loss, 2) if self.loss!=None else None}): '\
                   f'Prediction: {self.prediction}'
