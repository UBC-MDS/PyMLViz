from pymlviz.model_comparison_table import model_comparison_table

# import numpy and pandas
import pandas as pd

# import models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# import syn data generation
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

# synthetic data classification
syn_data_cf = make_classification(n_samples=1000, n_classes=4,
                                  n_informative=12)

# synthetic data regression
syn_data_reg = make_regression(n_samples=1000, n_informative=12)

# train test split classification data
tts_cf = train_test_split(pd.DataFrame(syn_data_cf[0]),
                          syn_data_cf[1],
                          test_size=0.33, random_state=42)

X_train_cf, X_test_cf, y_train_cf, y_test_cf = tts_cf

# train test split classification data
tts_reg = train_test_split(pd.DataFrame(syn_data_reg[0]),
                           syn_data_reg[1],
                           test_size=0.33, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = tts_reg

# fit classification models
lr_cf = LogisticRegression().fit(X_train_cf, y_train_cf)
svm_cf = SVC().fit(X_train_cf, y_train_cf)

# fit regression models
lr_reg = LinearRegression().fit(X_train_reg, y_train_reg)
svm_reg = SVR().fit(X_train_reg, y_train_reg)


# test normal functionality
def test_normal_function():
    """
    Test the output for regression
    and classification models
    are dataframes.
    """
    # output table from function for classification models
    op_table_cf = model_comparison_table(X_train_cf, y_train_cf,
                                         X_test_cf, y_test_cf,
                                         svm_model=svm_cf, lr_model=lr_cf)

    # test cf output
    assert isinstance(op_table_cf, pd.DataFrame), \
        "input table should be pd.DataFrame"

    # test reg output
    assert isinstance(op_table_reg, pd.DataFrame), \
        "input table should be pd.DataFrame"


def test_score_col():
    """
    The score can only be positive,
    needs to check output of the
    score columns.
    """
    # output table from function for classification models
    op_table_cf = model_comparison_table(X_train_cf, y_train_cf,
                                         X_test_cf, y_test_cf,
                                         svm_model=svm_cf, lr_model=lr_cf)

    # output table from function for regression models
    op_table_reg = model_comparison_table(X_train_reg, y_train_reg,
                                          X_test_reg, y_test_reg,
                                          svm_model=svm_reg, lr_model=lr_reg)

    assert op_table_cf["train_score"].min() >= 0, \
        "Classification train score should be >= 0"
    assert op_table_cf["test_score"].min() >= 0, \
        "Classification test score should be >= 0"


def check_homogenous_models():
    """
    Check whether all models
    are regression or
    classification models.
    """
    # all classification models, return dataframe.
    op_cf = model_comparison_table(X_train_cf, y_train_cf,
                                   X_test_cf, y_test_cf,
                                   svm_model=svm_cf, lr_model=lr_cf)

    op_reg = model_comparison_table(X_train_reg, y_train_reg,
                                    X_test_reg, y_test_reg,
                                    svm_model=svm_reg, lr_model=lr_reg)

    # all classification models, return dataframe.
    assert isinstance(op_cf, pd.DataFrame), \
        "result table should be pd.DataFrame"

    # all regression models, return dataframe.
    assert isinstance(op_reg, pd.DataFrame), \
        "result table should be pd.DataFrame"

    # function should not return a dataframe in the case regression and
    # classification models are passed
    bad_ip = model_comparison_table(X_train_reg, y_train_reg,
                                    X_test_reg, y_test_reg,
                                    svm_model=svm_cf, lr_model=lr_reg)
    assert not isinstance(bad_ip, pd.DataFrame), \
        "can not pass in both regression and classification models"
