import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC

from pymlviz.plot_confusion_m import plot_confusion_m

cancer = datasets.load_breast_cancer()
X = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=111)

svm = SVC()
svm.fit(X_train, y_train)


# Test the output type
def test_plot_output_type():
    confusion_plot = plot_confusion_m(svm, X_test, y_test)

    assert (str(type(confusion_plot)) ==
            "<class 'matplotlib.figure.Figure'>"), "Wrong plot output type"


# Test the input type
def test_input_type():
    try:
        plot_confusion_m(svm, [1, 2, 3, 4], y_test)
    except Exception as e:
        assert str(e) == "Sorry, X_test should " \
                         "be a pd.DataFrame or np.ndarray."

    try:
        plot_confusion_m(svm, X_test, [1, 2, 3, 4])
    except Exception as e:
        assert str(e) == "Sorry, y_test should be a np.ndarray."

    try:
        plot_confusion_m(svm, np.array([1, 2, '3']), y_test)
    except Exception as e:
        assert str(e) == "Sorry, all elements in X_test should be numeric."

    try:
        plot_confusion_m(svm, X_test, np.array([1, 2, '3']))
    except Exception as e:
        assert str(e) == "Sorry, all elements in y_valid should be numeric."

    try:
        plot_confusion_m(svm, np.array([4, 5, 6, 7]), np.array([1, 2, 3]))
    except Exception as e:
        assert str(e) == "Sorry, X_test and y_test" \
                         " should have the same number of rows."
