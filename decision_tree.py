#!/usr/bin/env python

import numpy as np
import os.path
import pandas
import sqlite3

# LoadData: Loads data from the specified database
def LoadData(DB_PATH, TRAINING_DATA=False):

    # Make sure that the database file exists
    assert os.path.isfile(DB_PATH), "Could not find database file at: {}".format(DB_PATH)

    # Establish a connection to the sqlite database
    connection = sqlite3.connect(DB_PATH)
    
    # Load the input into a Pandas DataFrame object for manipulation
    dataframe = pandas.read_sql_query("SELECT * FROM Customer;", connection)
    
    # Check how many rows and columns are in the loaded data
    error_message = "Unexpected input data shape: {}".format(dataframe.shape)
    assert (not TRAINING_DATA) or dataframe.shape == (14, 6), error_message
    assert TRAINING_DATA or dataframe.shape == (6, 6), error_message    
    print(dataframe)
    return dataframe

# Node: the data type used to build the decision tree model
class Node():

    # Constructor: Defines one of the following
    # - leaf_label: The label returned by a leaf node
    # - node_tuple: The definition of a non-leaf node, including
    #    - feature: The feature to split on
    #    - split:   The set of values leading to the left child
    #    - lhs:     The left-hand child
    #    - rhs:     The right-hand child
    def __init__(self, leaf_label=None, node_tuple=None):
        if node_tuple:
            assert not leaf_label, "Leaf label defined for non-leaf node"
            self.is_leaf = False
            self.feature, self.split, self.lhs, self.rhs = node_tuple
        else:
            self.is_leaf = True
            if leaf_label == None:
                self.leaf_label = "INVALID"
            else :
                self.leaf_label = leaf_label

    # Format the tree recursively
    def Format(self, level=0):
        indent = "- "
        for i in range(0, level):
            indent = "   " + indent
        if self.is_leaf:
            return indent + str(self.leaf_label)
        else:
            result = indent + self.feature + "({})".format(self.split)
            result += "\n" + self.lhs.Format(level+1)
            result += "\n" + self.rhs.Format(level+1)
            return result

    # Predict the classification label of a new data point
    def Evaluate(self, data_point):
        
        # TODO: Traverse the tree to predict a label for the data point
        node = self
        while not node.is_leaf:
            if data_point[node.feature] in node.split:
                node = node.lhs
            else:
                node = node.rhs
        label = node.leaf_label
        return label

# Calculate the impurity of the partition
def CalculateGiniImpurity(partition, label="buys_computer"):

    # TODO: Calculate the impurity of the partition (see equation 8.7 from the text)
    #       Do NOT use iteration or recursion in your calculations
    if len(partition) == 0:
        return 0
    total_count = len(partition)
    positive_count = np.sum(partition[label] == 1)
    negative_count = total_count - positive_count
    p_positive = positive_count / total_count
    p_negative = negative_count / total_count
    result = 1 - (p_positive**2 + p_negative**2)

    return result
    
# Calculate the Gini Index for a partition of the dataframe given the feature and split
def CalculateGini(dataframe, feature, split):

    # TODO: Calculate the Gini index of the split (see equation 8.8 from the text)
    #       Do NOT use iteration or recursion in your calculations
    lhs_partition = dataframe[dataframe[feature].isin(split)]
    rhs_partition = dataframe[~dataframe[feature].isin(split)]
    left_impurity = CalculateGiniImpurity(lhs_partition)
    right_impurity = CalculateGiniImpurity(rhs_partition)
    weight_left = len(lhs_partition) / len(dataframe)
    weight_right = len(rhs_partition) / len(dataframe)
    result = weight_left * left_impurity + weight_right * right_impurity
    return result
    
# Train the model by learning the decision tree
def Train(dataframe, label="buys_computer"):

    # TODO: Build up the decision tree using the Node class
    if len(dataframe[label].unique()) == 1:
        return Node(leaf_label=dataframe[label].iloc[0])
    # This example simply splits on the first feature; modify to find the best split
    feature = label
    split_value = dataframe[feature].unique()[0]
    split_set = [split_value]
    left_df = dataframe[dataframe[feature].isin(split_set)]
    right_df = dataframe[~dataframe[feature].isin(split_set)]
    if len(left_df) == 0 or len(right_df) == 0:
        return Node(leaf_label=dataframe[label].mode()[0])
    lhs = Train(left_df, label)
    rhs = Train(right_df, label)
    model = Node(node_tuple=(feature, split_set, lhs, rhs))
    print(f'Model Feature: {label}')
    return model
# Calculate and print metrics on the classifier model performance
def CalculatePerformance(results, expected):

    # TODO: Calculate the following performance metrics
    #       Do NOT use iteration or recursion in your calculations
    results = np.array(results)
    expected = np.array(expected)

    TP = np.sum((results == 1) & (expected == 1))
    TN = np.sum((results == 0) & (expected == 0))
    FP = np.sum((results == 1) & (expected == 0))
    FN = np.sum((results == 0) & (expected == 1))

    Accuracy = (TP + TN) / len(results)
    ErrorRate = (FP + FN) / len(results)
    Sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    Specificity = TN / (TN + FP) if TN + FP != 0 else 0
    Precision = TP / (TP + FP) if TP + FP != 0 else 0
    # Write the model tree structure to file
    f = open("decision_tree_results.txt", "w")
    f.write("Results: " + str(results) + "\n")
    f.write("Expected: " + str(expected) + "\n")
    f.write("TP: " + str(TP) + "\n")
    f.write("TN: " + str(TN) + "\n")
    f.write("FP: " + str(FP) + "\n")
    f.write("FN: " + str(FN) + "\n")
    f.write("Accuracy: " + str(Accuracy) + "\n")
    f.write("ErrorRate: " + str(ErrorRate) + "\n")
    f.write("Sensitivity: " + str(Sensitivity) + "\n")
    f.write("Specificity: " + str(Specificity) + "\n")
    f.write("Precision: " + str(Precision) + "\n")
    f.close()

# The entry-point of this program
if __name__=="__main__":

    # Path where you downloaded the training data
    DATA_PATH = './all_electronics_training.db'
    dataframe = LoadData(DATA_PATH, TRAINING_DATA=True)

    # Unit tests for the Gini calculations
    RUN_UNIT_TESTS = True
    if (RUN_UNIT_TESTS):
        ut1 = CalculateGiniImpurity(dataframe)
        assert abs(ut1 - 0.4591836) < 0.00001, "Incorrect impurity calculation: {}".format(ut1)

        ut2 = CalculateGini(dataframe, "income", ["low", "medium"])
        assert abs(ut2 - 0.4428571) < 0.00001, "Incorrect Gini calculation: {}".format(ut2)        
        
    # Build the decision tree
    model = Train(dataframe)

    # Write the model tree structure to file
    f = open("decision_tree_structure.txt", "w")
    f.write(model.Format() + "\n")
    f.close()

    # Path where you downloaded the test data
    DATA_PATH = './all_electronics_test.db'
    dataframe = LoadData(DATA_PATH)

    # Evaluate the test data
    results=[]
    for idxi, point_i in dataframe.iterrows():
        results.append(model.Evaluate(point_i))
    expected = dataframe["buys_computer"].to_list()
    print(f'Results: {results}')
    print(f'Expected: {expected}')

    # Calculate and print metrics on the classifier model performance
    CalculatePerformance(np.array(results), np.array(expected))
