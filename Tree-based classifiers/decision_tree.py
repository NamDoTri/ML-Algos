import numpy as np
from sklearn.preprocessing import LabelEncoder as le
from tree_node import Node

class DecisionTreeClassifier:
    def __init(self, *, criterion='gini', splitter='best', max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_spit = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = 0
        self.n_features = 0

    def best_split(self, data, labels):
        '''
            labels of all samples passed to this node
            data should be a DataFrame, which has the following structure:
                data
                ---feature_1
                ---feature_2
                ---feature_3
        '''
        # if number of samples is less than two, dont split
        m = labels.size
        if m < 2: return None, None

        # number of samples in each class passed into this node
        num_parent = [np.sum(labels == c) for c in range(self.n_classes)]

        # Gini of the current node (to see if it should or shouldn't be split in case there's no split which gives a lower Gini than the current one)
        best_gini = 1.0 - sum(k**2/m for k in num_parent)
        best_index, best_threshold = None, None

        for idx in range(len(data.columns)): # loop through every feature
            # thresholds is a list of feature values, 
            thresholds, classes = zip(*   # the * indicates unzipping operation
                                    sorted( zip(data[:, idx], labels) ) # zip data with labels and sort it based on values of the feature in the current loop 
                                    )

            # num_left and num_right are 2 lists which have the same number of elements, each element corresponds to a class.
            num_left = [0] * self.n_classes # initialize a list of 0s with the length of number of classes
            num_right = num_parent.copy()

            for i in range(1, m): # loop through all the possible splits in a feature
                
                # if 2 consecutive points have the same value, don't split at this index
                if thresholds[i] == thresholds[i-1]: continue 

                # update the number of samples on the left and right of current splitting index
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # calculate the gini values of children nodes
                gini_left = 1.0 - sum(
                    (num_left[x] / i)**2 for x in range(self.n_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m-i))**2 for x in range(self.n_classes)
                )
                gini = (i*gini_left + (m-i)*gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_index = idx # index of the feature
                    best_threshold = (thresholds[i] + thresholds[i-1])/2 # set the threshold to be midpoint

        return best_index, best_threshold

    def grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y)
        )
        pass

    def fit(self, X, y):
        pass


