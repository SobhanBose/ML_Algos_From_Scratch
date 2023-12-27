import numpy as np
import pandas as pd
import random

class Node:
    def __init__(self, feature, threshold, left, right, info_gain):
        self.is_leaf = False
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain


class Leaf:
    def __init__(self, value):
        self.is_leaf = True
        self.value = value


class DecisionTree:
    def __init__(self, min_samples: int=3, max_depth: int=2, num_classes: int=2) -> None:
        """

        Parameters
        ----------
        min_samples: int, optional
            by default 3
        max_depth: int, optional
            by default 2
        num_classes: int, optional
            by default 2
        """
        
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_classes = num_classes


    def __split(self, dataset: np.ndarray, feature: int, split_thresh: float) -> tuple:
        """
        Splits data into left and right branches

        Parameters
        ----------
        X: np.ndarray
            Data
        indices: list
            Active Indices
        feature: int
            Index of feature for splitting

        Returns
        -------
        left_indices: list
        right_indices: list
        """

        left_dataset = []
        right_dataset = []

        for row in dataset:
            if row[feature] <= split_thresh:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        return np.array(left_dataset), np.array(right_dataset)
    

    def __get_entropy(self, y: np.ndarray) -> float:
        """
        Computes the entropy

        Parameters
        ----------
        y: np.ndarray
            Label

        Returns
        -------
        Entropy: float
        """

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -(np.log(probabilities)/np.log(self.num_classes)))
        
        return entropy
    

    def __get_information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Get Information Gain

        Parameters
        ----------
        parent: np.ndarray
            Parent Node
        left: np.ndarray
            Left Child
        right: np.ndarray
            Right Child

        Returns
        -------
        information_gain: float
        """

        w_left = len(left)/len(parent)
        w_right = len(right)/len(parent)

        entropy_left, entropy_right = self.__get_entropy(left), self.__get_entropy(right)

        weighted_entropy = w_left*entropy_left + w_right*entropy_right

        information_gain = self.__get_entropy(parent) - weighted_entropy

        return information_gain
    

    def __get_best_split(self, dataset: np.ndarray, feature_indices: list[int]) -> dict:
        """
        Get best split parameters

        Parameters
        ----------
        dataset: np.ndarray
        feature_indices: list[int]
            
        Returns
        -------
        best_split: dict
            keys: gain, feature, threshold, left, right
        """

        best_split = {'gain': -1, 'feature': None, 'split_thresh': None}
        

        for feature_indx in feature_indices:
            feature_values = dataset[:, feature_indx]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left, right = self.__split(dataset, feature_indx, threshold)
                if len(left) and len(right):
                    y, left_y, right_y = dataset[:, -1], left[:, -1], right[:, -1]
                    information_gain = self.__get_information_gain(y, left_y, right_y)
                    if information_gain > best_split["gain"]:
                            best_split["feature"] = feature_indx
                            best_split["split_thresh"] = threshold
                            best_split["left"] = left
                            best_split["right"] = right
                            best_split["gain"] = information_gain
        
        return best_split


    def __getFeatureSubset(self, df: np.ndarray, subset_size: int=2) -> np.ndarray:
        """
        Get subset of features

        Parameters
        ----------
        df : np.ndarray
            Dataset
        subset_size : int, optional
            Number of features to be selected, by default 2

        Returns
        -------
        np.ndarray
            Selected feature indices
        """
        feature_indx = random.sample(range(0, df.shape[1]-1), k=subset_size)
        return feature_indx
    

    def __calculate_leaf_value(self, y: np.ndarray) -> int:
        """
        Calculate Leaf Value

        Parameters
        ----------
        y: np.ndarray

        Returns
        -------
        leaf_value: int
        """

        y = list(y)
        return max(y, key=y.count)


    def __build_tree_recur(self, dataset: np.ndarray, depth: int=0) -> Node:
        """
        Build the decision tree recursively

        Parameters
        ----------
        dataset: np.ndarray
        depth: int, optional

        Returns
        -------
        root: Node
        """

        X, y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples and depth <= self.max_depth:
            feature_indices = self.__getFeatureSubset(X, subset_size=3)

            best_split = self.__get_best_split(dataset, feature_indices)
            if best_split["gain"] >= 0:
                left_subtree = self.__build_tree_recur(best_split["left"], depth+1)
                right_subtree = self.__build_tree_recur(best_split["right"], depth+1)
                
                return Node(best_split["feature"], best_split["split_thresh"], 
                            left_subtree, right_subtree, best_split["gain"])

        leaf_value = self.__calculate_leaf_value(y)
        return Leaf(leaf_value)
    

    def fit(self, dataset: pd.DataFrame) -> None:
        """
        Fit model to data

        Parameters
        ----------
        dataset: pd.DataFrame
            Training Data
        """

        self.root = self.__build_tree_recur(dataset.to_numpy())
    

    def make_prediction(self, X: np.ndarray, node: Node):
        if node.is_leaf: 
            return node.value
        else:
            feature = X[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(X, node.left)
            else:
                return self.make_prediction(X, node.right)
            
            
    def predict(self, X: np.array) -> np.array:
        """
        Make prediction

        Parameters
        ----------
        X: pd.DataFrame
            Data

        Returns
        -------
        predictions: np.array
            Output Label
        """
        
        y_hat = [self.make_prediction(x, self.root) for x in X]
        return np.array(y_hat)
    