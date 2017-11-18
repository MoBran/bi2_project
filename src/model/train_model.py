import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def confusion_matrix(y_test, y_pred):
    cm = sk_confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(data=cm, columns=[1,2,3], index=[1,2,3])
    cm.columns.name = 'Predicted label'
    cm.index.name = 'True label'
    return cm

def get_logistic_regression(X_train, X_test, y_train, y_test, statsmodel=False):
    if statsmodel:
        pass
    else:
        # Multiclass prediction of rating 1,2,3
        logistic_regression = LogisticRegression(multi_class="multinomial",solver="newton-cg")
        logistic_regression.fit(X_train,y_train)

        # Make predictions using the testing set
        rating_predictions = logistic_regression.predict(X_test)

        report_model_performance(logistic_regression, rating_predictions, X_test, y_test)

    return logistic_regression

def base_line(X_train, X_test, y_train, y_test):
    base_columns = ["speed","time","distance"]
    X_train = X_train[base_columns]
    X_test = X_test[base_columns]
    return get_logistic_regression(X_train, X_test, y_train, y_test)

def report_model_performance(model, predictions, X_test, ground_truth):
    # The coefficients
    feature_names = list(X_test.columns.values)
    coefficients = pd.DataFrame(model.coef_,columns=feature_names)
    print('\nCoefficients: \n', coefficients)
    # The mean squared error
    print("Mean Accuracy: ", model.score(X_test, ground_truth))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ground_truth, predictions))
    error_rate = (predictions != ground_truth).mean()
    print('error rate: %.2f' % error_rate)


    print("Prediction:\n",[i for i in predictions])
    print("True label:\n", [i for i in ground_truth])

    cm = confusion_matrix(ground_truth, predictions)
    print("\nConfusion Matrix:\n", cm)
