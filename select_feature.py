# coding=utf-8

'''
    XGBoost模型
'''
import os
import gc

import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

from config import INPUT_PATH, OUTPUT_PATH, SAMPLE_PATH, RANDOM_STATE, THREADS
from logger import logger

NUM_OF_FOLDS = 5
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 100

INPUT_PATH = SAMPLE_PATH

def xgb_train(train_x, valid_x, train_y, valid_y):
    def mse(preds, dmatrix):
        label = dmatrix.get_label()
        score = mean_squared_error(label, preds)
        return 'mse', score

    y_preds_fold = np.zeros((valid_x.shape[0], NUM_OF_FOLDS))

    skf = KFold(n_splits=NUM_OF_FOLDS, shuffle=True)
    for n_fold, (train_index, val_index) in enumerate(skf.split(train_x, train_y)):
        dtrain = xgb.DMatrix(train_x.iloc[train_index, :], label=train_y.iloc[train_index])
        dval = xgb.DMatrix(train_x.iloc[val_index, :], label=train_y.iloc[val_index])

        xgb_model = xgb.train(
            params, dtrain, num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, 'train'), (dval, 'val')], feval=mse,
            verbose_eval=100, early_stopping_rounds=100
        )

        del dtrain, dval
        gc.collect()

        y_preds_fold[:, n_fold] = xgb_model.predict(
            xgb.DMatrix(valid_x), ntree_limit=xgb_model.best_iteration
        )
    score = mean_squared_error(valid_y, y_preds_fold.mean(axis=1))
    return score

if __name__ == '__main__':
    train_data = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))

    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'gamma': 2,
        'min_child_weight': 2,
        'max_depth': 7,
        'lambda': 10,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.1,
        'alpha': 0,
        'lambda': 0.05,
        'seed': RANDOM_STATE,
        'nthread': THREADS
    }
    y_train = train_data['solution']
    train_data.drop(columns=['solution'], inplace=True)
    X_train = train_data

    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.333,
                                                          random_state=0)

    print('base:', 0.18618269519650138)
    res = {'base': 0.18618269519650138}

    for feat_name in X_train.columns:
        tmp_score = xgb_train(train_x.drop(columns=feat_name), valid_x.drop(columns=feat_name), train_y, valid_y)
        res[feat_name] = tmp_score
        if tmp_score > 0.18618269519650138:
            print(feat_name, ':', tmp_score, '--------- N')
        else:
            print(feat_name, ':', tmp_score, '--------- Y')

    print(res)