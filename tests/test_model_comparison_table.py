from pymlviz.model_comparison_table import model_comparison_table

# import numpy and pandas
import pandas as pd
import numpy as np

# import models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# import syn data generation
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

# import pytest
import pytest

# synthetic data classification
syn_data_cf = make_classification(n_samples=1000, n_classes=4, n_informative=12)

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
svm_cf = LinearRegression().fit(X_train_reg, y_train_reg)

# test normal functionality
def test_normal_function():
    

