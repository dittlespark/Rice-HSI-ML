from sklearn import svm
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from LightGBM_reg import *
import statsmodels.api as sm
import math

def run_svm(train_X, train_y, test_X, test_y_true):
    # 定义svm模型
    clf = svm.SVR()
    train_X, test_X = pca_feature_selection(train_X, train_y, test_X, 30)
    # 训练svm模型
    clf.fit(train_X, train_y)
    # 预测测试集结果
    test_y_pred = clf.predict(test_X)
    # 计算测试集r方
    r2 = r2_score(test_y_true, test_y_pred)
    # 返回测试集r方
    return r2

def Random_Frost(train_X, train_y, test_X, test_y_true):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a Random Forest Regressor object
    rf = RandomForestRegressor()

    # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2,
                                       random_state=42, n_jobs=-1)

    # 训练随机森林模型
    random_search.fit(train_X, train_y)
    # 预测测试集结果
    test_y_pred = random_search.predict(test_X)
    # 计算测试集r方
    r2 = r2_score(test_y_true, test_y_pred)
    # 返回测试集r方
    return r2

def linear_regression(X_train, y_train, X_test, y_test):
    # 定义线性回归模型
    model = LinearRegression()
    X_train, X_test = pca_feature_selection(X_train, y_train, X_test, 35)

    # 训练线性回归模型
    model.fit(X_train, y_train)
    # 预测测试集结果
    y_pred = model.predict(X_test)
    # 计算测试集r方
    r2 = r2_score(y_test, y_pred)
    # 返回测试集r方
    return r2


def stepwise_linear_regression(X, y,X_test,y_test,
                               initial_list=[],
                               threshold_in=0.01,
                               threshold_out=0.05,
                               verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        X = pd.DataFrame(X)
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_pval = new_pval.min()
        if min_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, min_pval))
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        max_pval = pvalues.max()
        if max_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, max_pval))
        if not changed:
            break

    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
    # print(X.columns)
    X_test = pd.DataFrame(X_test)
    # print(X_test.columns)
    y_pred = model.predict(sm.add_constant(pd.DataFrame(X_test[included])))
    r2 = r2_score(y_test, y_pred)
    print(r2)

    return included,r2


def stepwise_linear_regression_AIC(X, y, X_test, y_test, threshold_in=0.05, threshold_out=0.1):
    """
    Perform stepwise linear regression on the input data to select the best set of model features using AIC as a criterion.

    Parameters:
    X (numpy.ndarray): The training input data.
    y (numpy.ndarray): The training output data.
    X_test (numpy.ndarray): The test input data.
    y_test (numpy.ndarray): The test output data.
    threshold_in (float): The threshold for feature addition. Default is 0.05.
    threshold_out (float): The threshold for feature removal. Default is 0.1.

    Returns:
    (numpy.ndarray): The predicted output data for the test input data.
    """
    # Initialize the feature set and the best feature set
    features = set(range(X.shape[1]))
    best_features = None
    best_aic = float('inf')

    # Perform stepwise linear regression
    while True:
        # Fit the model with the current feature set
        X_current = X[:, list(features)]
        X_test_current = X_test[:, list(features)]
        model = sm.OLS(y, X_current).fit()

        # Calculate the AIC value for the current model
        aic = model.aic

        # Update the best feature set if the current model has a lower AIC value
        if aic < best_aic:
            best_aic = aic
            best_features = features.copy()

        # Find the p-values for the remaining features
        p_values = model.pvalues
        max_p_value = p_values.max()

        # If the maximum p-value is greater than the threshold_out, remove the corresponding feature
        if max_p_value > threshold_out and math.isnan(max_p_value) == False:
            features.remove(list(p_values).index(max_p_value))

        # Otherwise, if all p-values are less than the threshold_in, stop the loop
        elif all(p_values < threshold_in):
            break

        # Otherwise, add the feature with the smallest p-value
        else:
            if any(math.isnan(x) for x in p_values) == False:
                features.add(list(p_values).index(min(p_values)))

    # Fit the final model with the best feature set and return the predicted output data for the test input data
    X_best = X[:, list(best_features)]
    X_test_best = X_test[:, list(best_features)]
    model = sm.OLS(y, X_best).fit()
    y_pred = model.predict(X_test_best)
    r2 = r2_score(y_test, y_pred)
    return r2





