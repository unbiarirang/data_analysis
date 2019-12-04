import pandas as pd
import numpy as np
import random
import math
import collections
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import feature_scaler

class DecisionTree(object):
    """One decision tree"""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def search_value(self, X):
        """find specific leaf node"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif X[self.split_feature] <= self.split_value:
            return self.tree_left.search_value(X)
        else:
            return self.tree_right.search_value(X)

    def show(self):
        """show the tree structure in json format"""
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.show()
        right_info = self.tree_right.show()
        structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, max_features=None, subsample=0.8, random_state=None):
        """
        n_estimators:      the number of trees in the forest.
        max_depth:         the maximum depth of the tree.
                           -1 means no limited depth.
        min_samples_split: the minimum number of samples required
                           to split an internal node.
        min_samples_leaf:  the minimum number of samples required
                           to be at a leaf node.
        min_split_gain:    the minimum gain required
                           to split an internal node.
        max_features:      The number of features to consider
                           when looking for the best split.
                           If "sqrt" then max_features=sqrt(n_features).
                           If "log2" then max_features=log(n_features).
        subsample:         Row sampling ratio
        random_state:      The seed used by the random number generator
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state
        self.trees = dict()

    def fit(self, X, Y):
        """Build a forest of trees from the training set"""
        Y = Y.to_frame(name='y')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        if self.max_features == "sqrt":
            self.max_features = int(len(X.columns) ** 0.5)
        elif self.max_features == "log2":
            self.max_features = int(math.log(len(X.columns)))
        else:
            self.max_features = len(X.columns)

        # build trees
        for idx in tqdm(range(self.n_estimators)):
            print(("iter: " + str(idx+1)).center(80, '='))

            # Randomly select rows and columns
            random.seed(random_state_stages[idx])
            subset_index = random.sample(range(len(X)), int(self.subsample * len(X)))
            subcol_index = random.sample(X.columns.tolist(), self.max_features)
            X_copy = X.loc[subset_index, subcol_index].reset_index(drop=True)
            Y_copy = Y.loc[subset_index, :].reset_index(drop=True)

            tree = self._fit(X_copy, Y_copy, depth=0)
            self.trees[idx] = tree
            print(tree.show())

    def _fit(self, X, Y, depth):
        """Build a decision tree"""
        # check for a no split and stop splitting
        if len(Y['y'].unique()) <= 1 or len(X) <= self.min_samples_split:
            tree = DecisionTree()
            tree.leaf_value = self.leaf_value(Y['y'])
            return tree

        # check for max depth
        if depth >= self.max_depth:
            # stop splitting
            tree = DecisionTree()
            tree.leaf_value = self.leaf_value(Y['y'])
            return tree

        # split dataset
        best_split_feature, best_split_value, best_split_gain = \
            self.choose_best_feature(X, Y)
        left_X, right_X, left_Y, right_Y= \
            self.split_dataset(X, Y, best_split_feature, best_split_value)

        tree = DecisionTree()
        # check for left and right child
        if len(left_X) <= self.min_samples_leaf or \
                len(right_X) <= self.min_samples_leaf or \
                best_split_gain <= self.min_split_gain:
            tree.leaf_value = self.leaf_value(Y['y'])
            return tree

        # process left and right child
        tree.split_feature = best_split_feature
        tree.split_value = best_split_value
        tree.tree_left = self._fit(left_X, left_Y, depth+1)
        tree.tree_right = self._fit(right_X, right_Y, depth+1)
        return tree

    def choose_best_feature(self, X, Y):
        """Select the best split point for a dataset"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in X.columns:
            if len(X[feature].unique()) <= 100:
                unique_values = sorted(X[feature].unique().tolist())
            # if there are too many features, select 100 percentile values as the candidate split thresholds
            else:
                unique_values = np.unique([np.percentile(X[feature].dropna(), x) for x in np.linspace(0, 100, 100)])
            # choose the split point with the largest gain
            for split_value in unique_values:
                if np.isnan(split_value): continue
                left_Y = Y[X[feature] <= split_value]
                right_Y = Y[X[feature] > split_value]
                split_gain = self.gini_impurity(left_Y['y'], right_Y['y'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def leaf_value(Y):
        """Select the most frequent value as the leaf value"""
        label_counts = collections.Counter(Y)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def gini_impurity(left_Y, right_Y):
        """Calculate the Gini index for a split dataset"""
        split_gain = 0
        for Y in [left_Y, right_Y]:
            gini = 1
            label_counts = collections.Counter(Y)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(Y)
                gini -= prob ** 2
            split_gain += len(Y) * 1.0 / (len(left_Y) + len(right_Y)) * gini
        return split_gain

    @staticmethod
    def split_dataset(X, Y, split_feature, split_value):
        """
        Split the dataset into two parts based on
        split_feature and split_value
        """
        left_X = X[X[split_feature] <= split_value]
        left_Y = Y[X[split_feature] <= split_value]
        right_X = X[X[split_feature] > split_value]
        right_Y = Y[X[split_feature] > split_value]
        return left_X, right_X, left_Y, right_Y

    def predict(self, X):
        """
        Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates.
        """
        res = []
        for _, row in X.iterrows():
            pred_list = []
            for _, tree in self.trees.items():
                pred_list.append(tree.search_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv("classification_data.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.dropna(axis=0)
    # binary classification
    df.loc[df.y == 1, 'y'] = 0
    df.loc[df.y > 1, 'y'] = 1

    n_trees = 20
    n_folds = 5
    clf = RandomForestClassifier(n_estimators=n_trees,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 max_features="sqrt",
                                 subsample=0.8,
                                 random_state=66)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2)
    scores = []
    for train_index, test_index in tqdm(kf.split(df)):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        # train
        clf.fit(train_data.ix[:, :-1], train_data.ix[:, 'y'])
        accuracy = metrics.accuracy_score(test_data.ix[:, 'y'], clf.predict(test_data.ix[:, :-1]))
        print('>>accuracy:', accuracy)
        scores.append(accuracy)
    print('Mean Accuracy: %.3f%%' % (sum(scores)*100/float(len(scores))))
