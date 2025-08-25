"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=6):
        self.criterion =criterion
        self.max_depth =max_depth

    class Node:
        """Internal node or leaf of the decision tree."""
        def __init__(self, attribute=None, threshold=None, left=None, right=None, value=None):
            self.attribute =attribute     
            self.threshold =threshold      #threshold for real features (None if discrete)
            self.left =left             
            self.right =right              
            self.value =value      

        def is_leaf(self):
            return self.value is not None
        
    def leaf_value(self, y: pd.Series):
        """Compute the value to store at a leaf node."""
        if self.is_regression:
            # mean of all if it is regression
            return y.mean()
        else:
            # majority class for classification
            return y.value_counts().idxmax()

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        # stopping conditions
        if ((depth >= self.max_depth) or (len(y.unique()) == 1 or X.shape[1] == 0)):
            return self.Node(value=self.leaf_value(y))

        features =pd.Series(X.columns)
        best_attr,best_thresh =opt_split_attribute(X, y, 'mse' if self.is_regression else self.criterion,features)

        if best_attr is None:
            return self.Node(value=self.leaf_value(y))

        # split data
        X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_thresh)

        # incase split fails, make it leaf
        if len(y_left) == 0 or len(y_right) == 0:
            return self.Node(value=self.leaf_value(y))

        # building left and right subtree
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        return self.Node(attribute=best_attr, threshold=best_thresh,
                         left=left_child, right=right_child)
    


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        self.is_regression = check_ifreal(y)
        criterion_funcs = {
            "information_gain": information_gain,
            "gini_index": gini_index
        }
        self.criterion_fn = criterion_funcs[self.criterion]
        # build tree recursively
        self.root = self.build_tree(X, y, depth=0)
    
    def predict_one_sample(self, x: pd.Series, node: 'DecisionTree.Node'):
        """Traverse the tree for one sample."""
        if node.is_leaf():
            return node.value

        # discrete feature
        if node.threshold is None:
            if x[node.attribute] ==1:   # one-hot encoding for discrete features
                return self.predict_one_sample(x, node.left)
            else:
                return self.predict_one_sample(x, node.right)
        # real feature
        else:
            if x[node.attribute] <=node.threshold:
                return self.predict_one_sample(x, node.left)
            else:
                return self.predict_one_sample(x, node.right)
            


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        predictions = []
        for _, row in X.iterrows():
            predictions.append(self.predict_one_sample(row, self.root))
        return pd.Series(predictions, index=X.index)
    
    
    def plot_single_node(self, node: 'DecisionTree.Node', depth=0):
        indent = "    " * depth
        if node.is_leaf():
            print(f"{indent}Y: {node.value}")
        else:
           
            if node.threshold is None:
                print(f"{indent}?( {node.attribute} == 1 )")
            else:
                print(f"{indent}?( {node.attribute} <= {node.threshold:.3f} )")
            # left branch
            self.plot_single_node(node.left, depth + 1)
            # right branch
            print(f"{'    ' * depth}N:", end="")
            if node.right.is_leaf():
                print(f" {node.right.value}")
            else:
                print()
                self.plot_single_node(node.right, depth + 1)

    def _add_nodes(self, dot, node, parent=None, edge_label=""):
        node_id = str(id(node))
        if node.is_leaf():
            # Leaf node: show its value
            dot.node(node_id, label=f"Leaf: {node.value}", shape="box", style="filled", color="lightgrey")
        else:
            # Decision node: show attribute + threshold if real
            if node.threshold is None:
                label = f"{node.attribute} == 1"
            else:
                label = f"{node.attribute} <= {node.threshold:.3f}"
            dot.node(node_id, label=label, shape="ellipse", style="filled", color="lightblue")
        
        # Connect to parent
        if parent:
            dot.edge(parent, node_id, label=str(edge_label))
        
        # Add children recursively if not leaf
        if not node.is_leaf():
            self._add_nodes(dot, node.left, parent=node_id, edge_label="Yes")
            self._add_nodes(dot, node.right, parent=node_id, edge_label="No")



    def plot(self,name) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        dot =Digraph()
        filename="tree_diagram"+name
        view=False
        self._add_nodes(dot, self.root)
        dot.render(filename, view=view, format="png",cleanup=True)
