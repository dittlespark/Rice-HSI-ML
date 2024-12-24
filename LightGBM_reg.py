import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def pca_feature_selection(X_train, y_train, X_test, n_components):
    """
    Performs PCA feature selection on the input data and returns the transformed data.

    Parameters:
    X_train (numpy array): Input features for training
    y_train (numpy array): Target variable for training
    X_test (numpy array): Input features for testing
    n_components (int): Number of principal components to keep

    Returns:
    X_train_pca (numpy array): Transformed input features for training
    X_test_pca (numpy array): Transformed input features for testing
    """

    # Create a PCA object with the specified number of components
    pca = PCA(n_components=n_components)

    # Fit the PCA object to the training data and transform the data
    X_train_pca = pca.fit_transform(X_train)

    # Transform the test data using the fitted PCA object
    X_test_pca = pca.transform(X_test)

    # Print the explained variance ratio of each principal component

    # Create a LightGBM dataset object for the transformed training data
    lgb_train = lgb.Dataset(X_train_pca, label=y_train)

    # Create a LightGBM dataset object for the transformed test data
    lgb_test = lgb.Dataset(X_test_pca)

    return X_train_pca, X_test_pca


# 准备数据
def lightgbm_reg(train_X, train_y, test_X, test_y_true):

    # train_X, test_X = pca_feature_selection(train_X, train_y, test_X, 50)

    train_data = lgb.Dataset(train_X, label=train_y)
    test_data = lgb.Dataset(test_X, label=test_y_true)

    # 设置超参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 5,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # 训练模型
    num_round = 1000
    bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=50)

    # 预测并输出结果
    y_pred = bst.predict(test_X)
    r2 = r2_score(test_y_true, y_pred)
    return r2


def lightgbm_reg_search(train_X, train_y, test_X, test_y_true):

    # train_X, test_X = pca_feature_selection(train_X, train_y, test_X, 50)

    train_data = lgb.Dataset(train_X, label=train_y)
    test_data = lgb.Dataset(test_X, label=test_y_true)

    # 设置超参数
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2', 'l1'},
    #     'num_leaves': 5,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0
    # }

    # 自动调参
    gridParams = {
        'learning_rate': [0.005, 0.01, 0.05],
        'n_estimators': [40, 60, 80],
        'num_leaves': [4, 5, 6],
        'boosting_type': ['gbdt'],
        'objective': ['regression'],
        'metric': [list(('l2', 'l1'))],  # 更改此行
        'feature_fraction': [0.6, 0.7, 0.8],
        'bagging_fraction': [0.6, 0.7, 0.8],
        'bagging_freq': [2, 4, 6],
    }

    grid = GridSearchCV(lgb.LGBMRegressor(), gridParams, verbose=0, cv=4, n_jobs=-1)
    grid.fit(train_X, train_y)

    # 训练模型
    num_round = 1000
    bst = lgb.train(grid.best_params_, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=50)

    # 预测并输出结果
    y_pred = bst.predict(test_X)
    r2 = r2_score(test_y_true, y_pred)

    return r2, grid.best_params_


def lightgbm_reg_kfold(train_X, train_y):


    # 设置超参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 2,
        'verbose': 0
    }

    # r2_scores = cross_val_score(model, train_X, train_y, cv=kf, scoring='r2')
    # rmse_scores = cross_val_score(model, train_X, train_y, cv=kf, scoring='neg_mean_squared_error')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = kf.split(train_X, train_y)

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    # 新建一个数组，存储每折的预测结果
    cv_pred = np.zeros(train_X.shape[0])

    r2_all = []
    rmse_all = []

    for i, (train_idx, val_idx) in enumerate(splits):
        # 划分训练集和验证集
        X_train, X_val = train_X[train_idx], train_X[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_val, label=y_val)


        num_round = 1000
        bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds =50)

        # 预测并输出结果
        y_pred = bst.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        # 计算rmse
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        r2_all.append(r2)
        rmse_all.append(rmse)

        # 将每折的预测结果存储到数组中
        cv_pred[val_idx] = y_pred

    # 计算最终的r2和rmse

    return r2_all, rmse_all, cv_pred


def lightgbm_reg_search_kflod(train_X, train_y):

    # train_X, test_X = pca_feature_selection(train_X, train_y, test_X, 50)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = kf.split(train_X, train_y)

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    # 新建一个数组，存储每折的预测结果
    cv_pred = np.zeros(train_X.shape[0])

    r2_all = []
    rmse_all = []

    for i, (train_idx, val_idx) in enumerate(splits):
        # 划分训练集和验证集
        X_train, X_val = train_X[train_idx], train_X[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_val, label=y_val)

    # 自动调参
        gridParams = {
            'learning_rate': [0.005, 0.01, 0.05],
            'n_estimators': [40, 60, 80],
            'num_leaves': [4, 5, 6],
            'boosting_type': ['gbdt'],
            'objective': ['regression'],
            'metric': [list(('l2', 'l1'))],  # 更改此行
            'feature_fraction': [0.6, 0.7, 0.8],
            'bagging_fraction': [0.6, 0.7, 0.8],
            'bagging_freq': [2, 4, 6],
        }

        grid = GridSearchCV(lgb.LGBMRegressor(), gridParams, verbose=0, cv=4, n_jobs=-1)
        grid.fit(train_X, train_y)

    # 训练模型
        num_round = 1000
        bst = lgb.train(grid.best_params_, train_data, num_round, valid_sets=[test_data], early_stopping_rounds =50)

        # 预测并输出结果
        y_pred = bst.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        r2_all.append(r2)
        rmse_all.append(rmse)

        # 将每折的预测结果存储到数组中
        cv_pred[val_idx] = y_pred

    return r2_all, rmse_all, cv_pred