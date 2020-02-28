## pymlviz 

![](https://github.com/UBC-MDS/pymlviz/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/pymlviz/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/pymlviz) ![Release](https://github.com/UBC-MDS/pymlviz/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/pymlviz/badge/?version=latest)](https://pymlviz.readthedocs.io/en/latest/?badge=latest)

Visualization package for ML models. 

> This package contains four functions to allow users to conveniently plot various visualizations as well as compare performance of different classifier models. The four functions will perform the following tasks: 
> 1.  Compare the performance of various models 
> 2.  Plot the confusion matrix based on the input data
> 3.  Plot the ROC curve and calculate the AUC 
> 4.  Plot the trained and test accuracy from a fitted model

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
|function_3_name| input |out| desc.|
|fun4name|input |out| desc|

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://pymlviz.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
