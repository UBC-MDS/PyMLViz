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


def plot_confusion_matrix(model, X_train, X_test, y_train, y_test, predicted_y = None, labels = None, title = None):
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
    
    X_train : pd.DataFrame/np.ndarray
        Training dataset without labels.

    y_train : np.ndarray
        Training labels.

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

