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
        
    scoring : str
        Scoring criteria for models,
        by default uses sklearn's default.
    """


def plot_confusion_m(model, X_test, y_test, predicted_y = None, labels = None, title = None):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    
    
    """
    Takes in a trained model with X and y
    values to produce a confusion matrix 
    visual. If predicted_y array is passed in,
    other evaluation scoring metrics such as 
    Recall, and precision will also be produced.

    Parameters:    
    ------------
    model : model instance 
        A trained classifier

    X_test : pd.DataFrame/np.ndarray
        Test dataset without labels.

    y_test : np.ndarray
        Test labels.

    predicted_y : np.ndarray
        Predicted target values
    
    labels : list, default=None
        The labels of the confusion matrix

    title : String, default=None
        Title of the confusion matrix

    Returns:
    ------------
    display : matplotlib visual
    """
    

    if not isinstance(X_test, pd.DataFrame) and not isinstance(X_test, np.ndarray):
        raise Exception("Sorry, X_test should be a pd.DataFrame or np.ndarray.")

    if not isinstance(y_test, np.ndarray):
        raise Exception("Sorry, y_test should be a np.ndarray.")

    if (isinstance(X_test, pd.DataFrame) and not np.issubdtype(X_test.to_numpy().dtype, np.number)) or \
    (isinstance(X_test, np.ndarray) and  not np.issubdtype(X_test.dtype, np.number)):
        raise Exception("Sorry, all elements in X_test should be numeric.")

    if not np.issubdtype(y_test.dtype, np.number):
        raise Exception("Sorry, all elements in y_valid should be numeric.")
    
    if y_test.shape[0] != X_test.shape[0]:
        raise Exception("Sorry, X_test and y_test should have the same number of rows.")
    

    confusion_matrix = plot_confusion_matrix(model, X_test, y_test, 
                                             display_labels = labels, 
                                             cmap=plt.cm.Reds, 
                                             values_format = 'd')
    if(title == None):
        confusion_matrix.ax_.set_title('Confusion Matrix')
    else:
        confusion_matrix.ax_.set_title(title)
        
    return confusion_matrix.figure_


def plot_train_valid_acc(model_name, X_train, y_train, X_valid, y_valid, param_name, param_vec):
    """
    Takes in a model name, train/validation data sets, 
    a parameter name and a vector of parameter values 
    to try and then plots train/validation accuracies vs.
    parameter values.

    Parameters:
    ------------
    model_name : str 
        the machine learning model name

    X_train : pd.DataFrame/np.ndarray
        Training dataset without labels.

    y_train : np.ndarray
        Training labels.

    X_valid : pd.DataFrame/np.ndarray
        Validation dataset without labels.

    y_valid : np.ndarray
        Test labels.
        
    param_name : str
        the parameter name.

    param_vec : list
        the parameter values.

    Returns:
    ------------
    display : matplotlib visual
    """

def plot_roc(model, X_valid, y_valid):
    """
    Takes in a fitted model, must be a fitted binary classifier, train/validation data sets, 
    plot a ROC curve

    Parameters:
    ------------
    model : str 
        the fitted binary classifier

    X_valid : pd.DataFrame/np.ndarray
        Validation dataset without labels.

    y_valid : np.ndarray
        validation set with labels.

    Returns:
    ------------
    display : matplotlib visual
    """
    
    # probs = model.predict_proba(X_valid)
    # preds = probs[:,1]
    # fpr, tpr, threshold = metrics.roc_curve(y_valid, preds)
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.title('ROC curve')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
