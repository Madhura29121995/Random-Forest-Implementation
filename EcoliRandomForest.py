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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv('ecoli.data', names=["Sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "decision"])

    data["decision"].replace(["cp","im","imU","imS","imL","om","omL","pp"], [0,1,2,3,4,5,6,7], inplace = True)
    data['mcg'] = [int(item) for item in data['mcg']]
    data['gvh'] = [int(item) for item in data['gvh']]
    data['lip'] = [int(item) for item in data['lip']]
    data['chg'] = [int(item) for item in data['chg']]
    data['aac'] = [int(item) for item in data['aac']]
    data['alm1'] = [int(item) for item in data['alm1']]
    data['alm2'] = [int(item) for item in data['alm2']]

    print (data.dtypes)
    #data['mcg'] = data['mcg'].astype(int)
    #data['gvh'] = data['gvh'].astype(int)
    #data['lip'] = data['lip'].astype(int)
    #data['chg'] = data['chg'].astype(int)
    #data['aac'] = data['aac'].astype(int)
    #data['alm1'] = data['alm1'].astype(int)
    #data['alm2'] = data['alm2'].astype(int)
    #data['decision'] = data['decision'].astype(int)

    features = [
        'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'
    ]

    y=data['decision']
    X=data[features] 

    y_arr = y.to_numpy()
    X_arr = X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=10, max_depth=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #acc = accuracy(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)