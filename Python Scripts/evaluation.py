import data, training
import sklearn


def evaluation_report(y, y_hat):
    print('Balanced Accuracy Score: ', sklearn.metrics.balanced_accuracy_score(y, y_hat))
    print('\n')
    print(sklearn.metrics.classification_report(y, y_hat))
    CM = sklearn.metrics.confusion_matrix(y, y_hat)
    print(CM)
    tn, fp, fn, tp = CM.ravel()
    print('True Negative:', tn)
    print('False Negative:', fn)
    print('True Positive:', tp)
    print('False Positive:', fp)
    print('Recall: ', sklearn.metrics.recall_score(y, y_hat))
    print('AUROC Score: ', sklearn.metrics.roc_auc_score(y, y_hat))
    return sklearn.metrics.roc_auc_score(y, y_hat)
    # Positive is SEU


def perform_evaluation(filename, clf):
    x_train, y_train, x_test, y_test = data.get_data(filename)
    y_hat = clf.predict(x_test)
    auroc = evaluation_report(y_test, y_hat)
    return auroc
