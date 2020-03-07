import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    display : matplotlib
    """
    
    try:
        model.predict(X_valid)
    except:
        raise Exception("Sorry, please make sure model is a fitted model.") 
        
    try:
        model.predict_proba(X_valid)
    except:
        raise Exception("Sorry, please make sure the model argument probability = True.") 
        
    if not isinstance(X_valid, pd.DataFrame) and not isinstance(X_valid, np.ndarray):
        raise Exception("Sorry, X_valid should be a pd.DataFrame or np.ndarray.")
        
    if not isinstance(y_valid, np.ndarray):
        raise Exception("Sorry, y_valid should be a np.ndarray.")
        
    if y_valid.shape[0] != X_valid.shape[0]:
        raise Exception("Sorry, X_valid and y_valid should have the same number of rows.")

    probs = model.predict_proba(X_valid)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_valid, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('ROC curve', fontsize=18)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right', fontsize=12)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
