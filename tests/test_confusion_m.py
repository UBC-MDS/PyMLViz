import pandas as pd
import numpy as np
import pytest
import sys

from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from pymlviz.plot_confusion_m import plot_confusion_m


cancer = datasets.load_breast_cancer()
X = pd.DataFrame(data = cancer['data'], columns = cancer['feature_names'])
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=111)

svm = SVC()
svm.fit(X_train, y_train)

#### NEED TO ADD MORE TESTS 
def test_plot_output_type():
    confusion_plot = plot_confusion_m(svm, X_test, y_test)

    assert (str(type(confusion_plot)) == "<class 'matplotlib.figure.Figure'>"), "Wrong plot output type"