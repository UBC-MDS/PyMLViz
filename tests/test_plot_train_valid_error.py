"""
Created on March 3, 2020

@author: Fanli Zhou

This script tests the plot_train_valid_error function
in the pymlviz package.

plot_train_valid_error function takes in a model name, 
train/validation data sets, a parameter name and a vector 
of parameter values to try and then plots train/validation 
errors vs. parameter values.
"""

from pymlviz.plot_train_valid_error import plot_train_valid_error
import pandas as pd
import numpy as np
import altair as alt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# use the iris data for unittest
iris = load_iris()
iris_df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                       columns = iris['feature_names'] + ['target'])

X = iris_df.drop(columns = ['target'])
y = iris_df[['target']]

X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                      y.to_numpy().ravel(), 
                                                      test_size = 0.2,
                                                      random_state=123)
p = alt.Chart(iris_df).encode(
    alt.X('target:Q'),
    alt.Y('target:Q')
).mark_line()

def test_output_type():
    """
    Test that output is a altair plot
    """
    assert type(plot_train_valid_error('knn', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'n_neighbors', range(1, 50))) == type(p), "The return type of knn is not correct."

    assert type(plot_train_valid_error('decision tree', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'max_depth', [5, 10, 15, 20])) == type(p), "The return type of decision tree is not correct."

    assert type(plot_train_valid_error('svc', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'c', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])) == type(p), "The return type of svc is not correct."

    assert type(plot_train_valid_error('svc', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'gamma', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])) == type(p), "The return type of svc is not correct."

    assert type(plot_train_valid_error('logistic regression', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'c', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]))== type(p), "The return type of logistic regression is not correct."

    assert type(plot_train_valid_error('random forests', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'max_depth', [5, 10, 15, 20])) == type(p), "The return type of random forests is not correct."

    assert type(plot_train_valid_error('random forests', 
                                       X_train, y_train, 
                                       X_valid, y_valid, 
                                       'n_estimators', [5, 10, 15, 20])) == type(p), "The return type of random forests is not correct."


def test_input_type():
    """
    Test for error if input is of a wrong type
    """
    try:
        plot_train_valid_error('knn', 
                                list(X_train), y_train, 
                                X_valid, y_valid, 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, X_train should be a pd.DataFrame or np.ndarray."

    try:
        plot_train_valid_error('knn', 
                                X_train, list(y_train), 
                                X_valid, y_valid, 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, y_train should be a np.ndarray."

    try:
        plot_train_valid_error('knn', 
                                X_train, y_train, 
                                list(X_valid), y_valid, 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, X_valid should be a pd.DataFrame or np.ndarray."

    try:
        plot_train_valid_error('knn', 
                                X_train, y_train, 
                                X_valid, list(y_valid), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, y_valid should be a np.ndarray."

    
    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, '3']), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, all elements in X_train should be numeric."

    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, '3']), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, all elements in y_train should be numeric."
        
    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                np.array([1, 2, '3']), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, all elements in X_valid should be numeric."
        
    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, '3']), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, all elements in y_valid should be numeric."

    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', [1, 2, '3'])
    except Exception as e:
        assert str(e) == "Sorry, all elements in para_vec should be numeric."

    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', [1, 2, -1])
    except Exception as e:
        assert str(e) == "Sorry, all elements in para_vec should be non-negative."

    try:
        plot_train_valid_error('knn', 
                                np.array([[1, 2, 3], [1, 2, 3]]), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, X_train and y_train should have the same number of rows."

    try:
        plot_train_valid_error('knn', 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                np.array([[1, 2, 3], [1, 2, 3]]), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, X_valid and y_valid should have the same number of rows."

    try:
        plot_train_valid_error('knn', 
                                np.array([[1, 2], [2, 3], [3, 4]]), np.array([1, 2, 3]), 
                                np.array([1, 2, 3]), np.array([1, 2, 3]), 
                                'n_neighbors', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, X_train and X_valid should have the same number of columns."
        
        
def test_input_value():
    """
    Test for error if input is of a wrong value
    """
    try:
        plot_train_valid_error('knn', 
                                X_train, y_train, 
                                X_valid, y_valid, 
                                'n', range(1, 50))
    except Exception as e:
        assert str(e) == "Sorry, only the hyperparameter 'n_neighbors' is allowed for a 'KNN' model."
        
    try:
        plot_train_valid_error('decision tree', 
                               X_train, y_train, 
                               X_valid, y_valid, 
                               'm', [5, 10, 15, 20])
    except Exception as e:
        assert str(e) == "Sorry, only the hyperparameter 'max_depth' is allowed for a 'decision tree' model."
                
    try:
        plot_train_valid_error('svc', 
                               X_train, y_train, 
                               X_valid, y_valid, 
                               'c', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    except Exception as e:
        assert str(e) == "Sorry, only the hyperparameters, 'c' and 'gamma', are allowed for a 'svc' model."
        
    try:
        plot_train_valid_error('logistic regression', 
                               X_train, y_train, 
                               X_valid, y_valid, 
                               'cv', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    except Exception as e:
        assert str(e) == "Sorry, only the hyperparameter 'c' is allowed for a 'logistic regression' model."
        
    try:
        plot_train_valid_error('random forests', 
                               X_train, y_train, 
                               X_valid, y_valid, 
                               'cv', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    except Exception as e:
        assert str(e) == "Sorry, only the hyperparameters, 'max_depth' and 'n_estimators', are allowed for a 'random forests' model."
        
    try:
        plot_train_valid_error('r', 
                               X_train, y_train, 
                               X_valid, y_valid, 
                               'c', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    except Exception as e:
        assert str(e) == "Sorry, the model_name should be chosen from 'knn', 'decision tree', 'svc', 'logistic regression', and 'random forests'." 
        
        
       