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
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt

    probs = model.predict_proba(X_valid)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_valid, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()