def model_comparison_table(models, X_train, y_train, X_test, y_test, scoring="default"):
"""
Takes in a list of models and
the train test data then outputs
a table comparing the scores for 
different models.

Parameters:
------------
models : list
    A list containing different models
    that have to be compared. The list
    must contain models for same task,
    ie. all classification models or
    all regression models.

X_train : pd.DataFrame/np.ndarray
    Training dataset without labels.

y_train : np.ndarray
    Training labels.

X_test : pd.DataFrame/np.ndarray
    Test dataset without labels.

y_test : np.ndarray
    Test labels.
"""