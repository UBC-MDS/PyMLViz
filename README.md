## pymlviz 

![](https://github.com/UBC-MDS/pymlviz/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/pymlviz/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/pymlviz) ![Release](https://github.com/UBC-MDS/pymlviz/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/pymlviz/badge/?version=latest)](https://pymlviz.readthedocs.io/en/latest/?badge=latest)

Visualization package for ML models. 

> This package contains four functions to allow users to conveniently plot various visualizations as well as compare performance of different classifier models. The purpose of this package is to reduce time spent on developing visualizations and comparing models, to speed up the model creation process for data scientists. The four functions will perform the following tasks: 
> 1.  Compare the performance of various models 
> 2.  Plot the confusion matrix based on the input data
> 3.  Plot train/validation errors vs. parameter values
> 4.  Plot the ROC curve and calculate the AUC 

|Contributors|GitHub Handle|
|------------|-------------|
|Anas Muhammad| [anasm-17](https://github.com/anasm-17)|
|Tao Huang|[taohuang-ubc](https://github.com/taohuang-ubc)|
|Fanli Zhou|[flizhou](https://github.com/mikeymice)|
|Mike Chen|[miketianchen](https://github.com/miketianchen)|

### Installation:

```
pip install -i https://test.pypi.org/simple/ pymlviz
```

### Features
| Function Name | Input | Output | Description |
|-------------|-----|------|-----------|
|model_comparison_table| List of model, X_train, y_train, X_test, y_test, scoring option | Dataframe of model score| Takes in a list of models and the train test data then outputs a table comparing the scores for different models.|
|plot_confusion_matrix | Model, X_train, y_train, X_test, y_test, predicted_y  | Confusion Matrix Plot, Dataframe of various scores (Recall, F1 and etc)| Takes in a trained model with X and y values to produce a confusion matrix visual. If predicted_y array is passed in, other evaluation scoring metrics such as Recall, and precision will also be produced.|
|plot_train_valid_error| model_name, X_train, y_train, X_valid, y_valid, param_name, param_vec |Train/validation errors vs. parameter values plot| Takes in a model name, train/validation data sets, a parameter name and a vector of parameter values and then plots train/validation errors vs. parameter values.|
|plot_roc|model, X_valid, y_valid|ROC plot| Takes in a fitted model, the validation set(X_valid) and the validation set labels(y_valid) and plots the ROC curve. The ROC curve also produces AUC score.|

### Alignment with Python Ecosystems

For some of our functions, there are not existing packages that implement the exact same functionality in Python. Most of these functions helps to show insights about machine learning models conveniently.

The comparisons between new functions and existing functions are:

| New functions | Existing Packages/Functions |
|-------------|-----|
|plot_confusion_matrix| [Sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) | 


### Dependencies

- python 3.7.3 with packages:
  - pytest-cov >= 2.8
  - numpy >= 1.18
  - pandas >= 1.0
  - scikit-learn >= 0.22.2
  - altair == 4.0.0

### Usage

#### model_comparison_table()

```
from pymlviz.model_comparison_table import model_comparison_table
model_comparison_table(X_train_reg, y_train_reg, X_test_reg, y_test_reg,
                       lr_reg_mod = lr_reg, svm_reg_mod = svm_reg)
>>>     model_name  train_score  test_score
    0   lr_reg_mod          0.89         0.76
    1  svm_reg_mod          0.77         0.78
```
#### plot_roc()

```
from pymlviz.plot_roc import plot_roc
plot_roc(SVC(), X_valid, y_valid)
>>> <plot_output>
```

#### plot_train_valid_error()

```
from pymlviz.plot_train_valid_error import plot_train_valid_error
plot_train_valid_error('random forests', 
                       X_train, y_train, 
                       X_valid, y_valid, 
                       'n_estimators', [5, 10, 15, 20])
>>> <plot_output>
```

### plot_confusion_m()

```
from pymlviz.plot_confusion_m import plot_confusion_m
plot_confusion_m(svm, X_test, y_test)
>>> <plot_output>
```

### Documentation
The official documentation is hosted on Read the Docs: <https://pymlviz.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
