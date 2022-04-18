from collections import Counter
import numpy as np
import pandas as pd
from DecisionTreeRandomForest import DecisionTree

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X_arr, y_arr):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X_arr, y_arr)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import accuracy_score

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv('letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])

    print (data.dtypes)
    ObjectColumns = data.select_dtypes(include=np.object).columns.tolist()
    data['lettr'] = [ord(item)-64 for item in data['lettr']]
    print(data["lettr"].iloc[23])

    features = ['x-box', 
    'y-box', 
    'width', 
    'high', 
    'onpix', 
    'x-bar', 
    'y-bar', 
    'x2bar', 
    'y2bar', 
    'xybar', 
    'x2ybr', 
    'xy2br', 
    'x-ege', 
    'xegvy', 
    'y-ege', 
    'yegvx'
    ]

    acc = []
    for i in range(10):
        y=data['lettr']
        X=data[features] 

        y = y.to_numpy()
        X = X.to_numpy()
 
        kf = KFold (n_splits= 5)
        for train_index, test_index in kf.split(X):
            X_train , X_test = X[train_index,:],X[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
        
            clf = RandomForest(n_trees=20, max_depth=10)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
            acc.append(accuracy_score(y_pred , y_test))
        
        data = data.sample(frac=1)


    print("Accuracy:", np.mean(acc))