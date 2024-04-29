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
        label="INVALID"
        
        return label

# Calculate the impurity of the partition
def CalculateGiniImpurity(partition, label="buys_computer"):

    # TODO: Calculate the impurity of the partition (see equation 8.7 from the text)
    #       Do NOT use iteration or recursion in your calculations
    result=0.0
    
    return result
    
# Calculate the Gini Index for a partition of the dataframe given the feature and split
def CalculateGini(dataframe, feature, split):

    # TODO: Calculate the Gini index of the split (see equation 8.8 from the text)
    #       Do NOT use iteration or recursion in your calculations
    result = 0.0

    return result
    
# Train the model by learning the decision tree
def Train(dataframe, label="buys_computer"):

    # TODO: Build up the decision tree using the Node class
    model = None
    return model

# Calculate and print metrics on the classifier model performance
def CalculatePerformance(results, expected):

    # TODO: Calculate the following performance metrics
    #       Do NOT use iteration or recursion in your calculations
    TP = None
    TN = None
    FP = None
    FN = None
    Accuracy = None
    ErrorRate = None
    Sensitivity = None
    Specificity = None
    Precision = None
    
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

    # Calculate and print metrics on the classifier model performance
    CalculatePerformance(np.array(results), np.array(expected))
