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

    data = pd.read_csv('mushroom.data', names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment", 
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", 
    "population", "habitat"])

    data["decision"].replace(["e", "p"], [0, 1], inplace = True)
    data["cap-shape"].replace(["b", "c", "x", "f", "k", "s"], [0, 1, 2, 3, 4, 5], inplace = True)
    data["cap-surface"].replace(["f", "g", "y", "s"], [0, 1, 2, 3], inplace = True)
    data["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
    data["bruises"].replace(["t", "f"], [0, 1], inplace = True)
    data["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
    data["gill-attachment"].replace(["a", "d", "f", "n"], [0, 1, 2, 3], inplace = True)
    data["gill-spacing"].replace(["c", "w", "d"], [0, 1, 2], inplace = True)
    data["gill-size"].replace(["b", "n"], [0, 1], inplace = True)
    data["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace = True)
    data["stalk-shape"].replace(["e", "t"], [0, 1], inplace = True)
    data["stalk-root"].replace(["b", "c", "u", "e", "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace = True)
    data["stalk-surface-above-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace = True)
    data["stalk-surface-below-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace = True)
    data["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
    data["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
    data["veil-type"].replace(["p", "u"], [1, 2], inplace = True)
    data["veil-color"].replace(["n", "o", "w", "y"], [1, 2, 3, 4], inplace = True)
    data["ring-number"].replace(["n", "o", "t"], [1, 2, 3], inplace = True)
    data["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [1, 2, 3, 4, 5, 6, 7, 8], inplace = True)
    data["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
    data["population"].replace(["a", "c", "n", "s", "v", "y"], [1, 2, 3, 4, 5, 6], inplace = True)
    data["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [1, 2, 3, 4, 5, 6, 7], inplace = True)

    data["  1"] = data['decision'].astype(int)
    data["cap-shape"] = data['cap-shape'].astype(int)
    data["cap-surface"] = data['cap-surface'].astype(int)
    data["cap-color"] = data['cap-color'].astype(int)
    data["bruises"] = data['bruises'].astype(int)
    data["odor"] = data['odor'].astype(int)
    data["gill-attachment"] = data['gill-attachment'].astype(int)
    data["gill-spacing"] = data['gill-spacing'].astype(int)
    data["gill-size"] = data['gill-size'].astype(int)
    data["gill-color"] = data['gill-color'].astype(int)
    data["stalk-shape"] = data['stalk-shape'].astype(int)
    data["stalk-root"] = data['stalk-root'].astype(int)
    data["stalk-surface-above-ring"] = data['stalk-surface-above-ring'].astype(int)
    data["stalk-surface-below-ring"] = data['stalk-surface-below-ring'].astype(int)
    data["stalk-color-above-ring"] = data['stalk-color-above-ring'].astype(int)
    data["stalk-color-below-ring"] = data['stalk-color-below-ring'].astype(int)
    data["veil-type"] = data['veil-type'].astype(int)
    data["veil-color"] = data['veil-color'].astype(int)
    data["ring-number"] = data['ring-number'].astype(int)
    data["ring-type"] = data['ring-type'].astype(int)
    data["spore-print-color"] = data['spore-print-color'].astype(int)
    data["population"] = data['population'].astype(int)
    data["habitat"] = data['habitat'].astype(int)

    features = [
        'cap-shape', 
        'cap-surface', 
        'cap-color', 
        'bruises', 
        'odor', 
        'gill-attachment', 
        'gill-spacing', 
        'gill-size', 
        'gill-color', 
        'stalk-shape', 
        'stalk-root', 
        'stalk-surface-above-ring', 
        'stalk-surface-below-ring', 
        'stalk-color-above-ring', 
        'stalk-color-below-ring', 
        'veil-type', 
        'veil-color', 
        'ring-number', 
        'ring-type', 
        'spore-print-color', 
        'population', 
        'habitat'
    ]

    acc = []
    for i in range(10):
        y=data['decision']
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