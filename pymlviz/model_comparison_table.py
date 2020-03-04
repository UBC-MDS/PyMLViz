import numpy as np
import pandas as pd

def model_comparison_table(X_train, y_train, X_test, y_test, **kwargs):
    """
    Takes in scikit learn ML models and
    the train test data then outputs
    a table comparing the scores for 
    different models.

    Parameters:
    ------------
    X_train : pd.DataFrame/np.ndarray
        Training dataset without labels.

    y_train : np.ndarray
        Training labels.

    X_test : pd.DataFrame/np.ndarray
        Test dataset without labels.

    y_test : np.ndarray
        Test labels.
        
    **kwargs : 
        Models assigned with meaningful
        variable names.

    Returns:
    --------
    pd.DataFrame
        Dataframe object consisting of
        models and comparison metrics.
    
    Example:
    --------
    model_comparison_table(X_train, y_train, X_test, y_test, 
        svc_model=svc_trained, lr_model=lr_trained)
    """
    # create dataframe skeleton for model
    df_results = pd.DataFrame({"model_name": [],
                             "train_score": [],
                             "test_score": []})
    
    # loop through models specified by user
    for model in kwargs:
        # compute values for results table
        train_score = kwargs[model].score(X_train, y_train)
        test_score = kwargs[model].score(X_test, y_test)
        model_name = model
        
        # create temporary results table
        df_res = pd.DataFrame({"model_name": [model_name],
                             "train_score": [train_score],
                             "test_score": [test_score]})
        
        # update results table
        df_results = df_results.append(df_res, ignore_index=True)
    
    # return dataframe
    return df_results