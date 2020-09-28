# coding=utf-8

'''
    LightGBM模型
'''
import os
import gc
from datetime import datetime

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

from config import INPUT_PATH, OUTPUT_PATH, SAMPLE_PATH, RANDOM_STATE, THREADS
from logger import logger
from utils import lgb_int_mse

NUM_OF_FOLDS = 5
NUM_BOOST_ROUND = 50000
EARLY_STOPPING_ROUNDS = 100

# INPUT_PATH = SAMPLE_PATH

if __name__ == '__main__':
    logger.info('Train LightGBM model...')
    start_time = datetime.now()

    used_feature = ['#followers', '#friends', '#favorites', 'day', 'hour', 'sentiment_left',
                    'sentiment_sum', 'sentiment_diff', 'entities_min', 'entities_len',
                    'mentions_len', 'mentions_value', 'hashtags_len', 'hashtags_value',
                    'usercount', 'username_lbl', 'entities_lbl', 'mentions_lbl', 'hashtags_lbl',
                    'urls_lbl', 'inner_favorite']

    train_data = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))[used_feature + ['solution']]
    logger.info('Load ' + str(os.path.join(INPUT_PATH, 'train.csv')))
    X_valid = pd.read_csv(os.path.join(INPUT_PATH, 'X_valid.csv'))[used_feature]
    logger.info('Load ' + str(os.path.join(INPUT_PATH, 'X_valid.csv')))
    X_test = pd.read_csv(os.path.join(INPUT_PATH, 'X_test.csv'))[used_feature]
    logger.info('Load ' + str(os.path.join(INPUT_PATH, 'X_test.csv')))

    params = {
        'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'rmse', 'mse'}, 'learning_rate': 0.08,
        'seed': RANDOM_STATE, 'num_threads': THREADS, 'verbose': 1
    }
    logger.info('The LightGBM model params: ' + str(params))

    y_train = train_data['solution']
    train_data.drop(columns=['solution'], inplace=True)
    X_train = train_data

    y_valid_preds_fold = np.zeros((X_valid.shape[0], NUM_OF_FOLDS))
    y_test_preds_fold = np.zeros((X_test.shape[0], NUM_OF_FOLDS))

    lgb_model = ''
    skf = KFold(n_splits=NUM_OF_FOLDS, shuffle=True)
    for n_fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        logger.info('Start %d-fold training...' % (n_fold))
        lgb_train = lgb.Dataset(X_train.iloc[train_index, :], label=y_train.iloc[train_index])
        lgb_valid = lgb.Dataset(X_train.iloc[val_index, :], label=y_train.iloc[val_index], reference=lgb_train)
        logger.info('%d-fold train shape: (%d, %d), the valid shape: (%d, %d)' % (n_fold, train_index.shape[0], X_train.shape[1], val_index.shape[0], X_train.shape[1]))

        lgb_model = lgb.train(
            params, lgb_train, num_boost_round=NUM_BOOST_ROUND, valid_sets=[lgb_train, lgb_valid], feval=lgb_int_mse,
            valid_names=['train', 'valid'], verbose_eval=100, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        lgb_model.save_model('lgb_model_'+str(n_fold)+'.m')

        del lgb_train, lgb_valid
        gc.collect()

        logger.info(
            '%d-fold best score is %f and number of trees is %d' % (
                n_fold, lgb_model.best_score['valid']['l2'], lgb_model.best_iteration
            )
        )

        y_valid_preds_fold[:, n_fold] = lgb_model.predict(X_valid)
        y_test_preds_fold[:, n_fold] = lgb_model.predict(X_test)

    y_valid_preds = y_valid_preds_fold.mean(axis=1)
    y_test_preds = y_test_preds_fold.mean(axis=1)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    datetime_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    with open(os.path.join(OUTPUT_PATH, 'validation_' + datetime_now + '.predict'), 'w') as fout:
        for each in y_valid_preds:
            if each < 0:
                fout.write('0\n')
            else:
                fout.write(str(int(np.round(np.exp(each) - 1))) + '\n')

    with open(os.path.join(OUTPUT_PATH, 'test_' + datetime_now + '.predict'), 'w') as fout:
        for each in y_test_preds:
            if each < 0:
                fout.write('0\n')
            else:
                fout.write(str(int(np.round(np.exp(each) - 1))) + '\n')

    end_time = datetime.now()
    logger.info('LightGBM over. Cost ' + str(end_time - start_time))

