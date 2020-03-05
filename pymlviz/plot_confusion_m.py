from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_m(model, X_test, y_test, predicted_y = None, labels = None, title = None):
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