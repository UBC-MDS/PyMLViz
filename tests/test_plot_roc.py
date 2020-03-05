"""
Created on Mar 3 10:05:13 2020
@author: Tao Huang
This script tests the plot_roc function of the pymlviz package.

The plot_roc function returns a ROC curve (with a AUC score) 
for a fitted model and its train and validation set.
"""

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from pymlviz.plot_roc import plot_roc
from sklearn.svm import SVC

# Sample input
breast_cancer  = load_breast_cancer(return_X_y = True)
X,y = breast_cancer
X_train_breast = X[:400]
y_train_breast = y[:400]
X_valid_breast = X[400:569]
y_valid_breast = y[400:569]
svc = SVC(probability=True)
svc.fit(X_train_breast,y_train_breast)

def test_output_plot_number():
    """
    Test that one plot is produced
    """
    plot_roc(svc, X_valid_breast, y_valid_breast)
    plt.gcf().number
    assert plt.gcf().number == 1

