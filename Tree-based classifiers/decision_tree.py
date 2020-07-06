import numpy as np
from sklearn.preprocessing import LabelEncoder as le

class DecisionTreeClassifier():
    def __init(self, *, criterion='gini', splitter='best', max_depth=None, min_smaples_split=None, min_samples_leaf=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_spit = min_smaples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, data):
        self.labels = data.labels.unique()
        if self.criterion == 'gini':
            if not (self.max_depth is None):
                for layer in range(self.max_depth):
                    pass
            else:
                pass

    def calc_gini(self, data, labels):
        pass

    def transform_data(self, raw_data):
        pass

print("something")