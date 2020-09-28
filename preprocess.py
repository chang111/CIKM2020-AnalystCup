# coding=utf-8

'''
    数据预处理
'''
import os
import gc

import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from config import INPUT_PATH, SAMPLE_PATH
from logger import logger

# INPUT_PATH = SAMPLE_PATH

if __name__ == '__main__':
    logger.info('Start preprocess...')

    # get the column
    feature_name = pd.read_table(os.path.join(INPUT_PATH, 'feature.name'), sep='\t')
    logger.info('Read ' + str(os.path.join(INPUT_PATH, 'feature.name')))
    
    # read train.data
    X_train_list = []
    with open(os.path.join(INPUT_PATH, 'train.data'), 'r') as fin:
        for line in fin:
            X_train_list.append(line.strip('\n').split('\t'))
    logger.info('Read ' + str(os.path.join(INPUT_PATH, 'train.data')))

    # read train.solution
    y_train_list = []
    with open(os.path.join(INPUT_PATH, 'train.solution'), 'r') as fin:
        for line in fin:
            y_train_list.append([line.strip('\n')])
    logger.info('Read ' + str(os.path.join(INPUT_PATH, 'train.solution')))

    # read validation.data
    X_valid_list = []
    with open(os.path.join(INPUT_PATH, 'validation.data'), 'r') as fin:
        for line in fin:
            X_valid_list.append(line.strip('\n').split('\t'))
    logger.info('Read ' + str(os.path.join(INPUT_PATH, 'validation.data')))

    # read test.data
    X_test_list = []
    with open(os.path.join(INPUT_PATH, 'test.data'), 'r') as fin:
        for line in fin:
            X_test_list.append([0] + line.strip('\n').split('\t'))
    logger.info('Read ' + str(os.path.join(INPUT_PATH, 'test.data')))

    X_data = pd.DataFrame(X_train_list + X_valid_list + X_test_list, columns=feature_name.columns.tolist())
    y_train = np.array(y_train_list).astype(np.float32)
    num_of_train, num_of_valid, num_of_test = len(X_train_list), len(X_valid_list), len(X_test_list)

    del X_train_list, X_valid_list, X_test_list, y_train_list
    gc.collect()

    # others
    X_data['#followers'] = X_data['#followers'].astype(int)
    X_data['#friends'] = X_data['#friends'].astype(int)
    X_data['#favorites'] = X_data['#favorites'].astype(int)

    # #followers, #friends, #favorites, inner_favorite
    X_data['inner_favorite'] = X_data['#favorites'].map(lambda x: x - int(x / 7) * 5)
    # num_features = ['#followers', '#friends', '#favorites', 'inner_favorite']
    # skewed_feats = X_data[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # logger.info('Skew in numerical features:')
    # skewness = pd.DataFrame({'Skew': skewed_feats})
    # skewness = skewness[abs(skewness) > 0.75]
    # logger.info('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
    # skewed_features = skewness.index
    # lam = 0.15
    # for feat in skewed_features:
    #     X_data[feat] = boxcox1p(X_data[feat], lam)

    # Timestamp
    start_day = datetime.strptime('2019-09-30', '%Y-%m-%d')
    timestamp = [datetime.strptime(tim, '%a %b %d %H:%M:%S +0000 %Y') for tim in X_data['timestamp']]
    X_data['year'] = [dat.year for dat in timestamp]
    X_data['month'] = [dat.month for dat in timestamp]
    # X_data['weekday'] = [dat.weekday() for dat in timestamp]
    X_data['day'] = [dat.day for dat in timestamp]
    X_data['hour'] = [dat.hour for dat in timestamp]
    X_data['timestamp'] = [(dat - start_day).days for dat in timestamp]
    logger.info('Create feature: timestamp, month, day, hour, year.')

    # labelencoder
    for col in ('username', 'entities', 'mentions', 'hashtags', 'urls'):
        lbl = LabelEncoder()
        X_data[col+'_lbl'] = lbl.fit_transform(X_data[col])
        logger.info('label encoder: ' + col + '.')

    # username
    username_map = X_data['username'].value_counts().to_dict()
    X_data['usercount'] = X_data['username'].map(lambda x : username_map[x])
    logger.info('Create feature: usercount.')

    # sentiment
    X_data['sentiment_left'] = X_data['sentiment'].map(lambda x: int(x.split(' ')[0]))
    X_data['sentiment_right'] = X_data['sentiment'].map(lambda x: int(x.split(' ')[1]))
    X_data['sentiment_sum'] = X_data['sentiment_left'] + X_data['sentiment_right']
    # X_data['sentiment_sumabs'] = (X_data['sentiment_left'] + X_data['sentiment_right']).map(lambda x: abs(x))
    X_data['sentiment_diff'] = X_data['sentiment_left'] - X_data['sentiment_right']
    # X_data['sentiment_div'] = X_data['sentiment_left'] / X_data['sentiment_right']
    logger.info('Create feature: sentiment_left, sentiment_right, sentiment_sum, sentiment_diff.')

    # entities
    # entities_mean = []
    # for each_entity in X_data['entities']:
    #     if each_entity != 'null;':
    #         tmp = []
    #         for each in each_entity.strip(';').split(';'):
    #             tmp.append(float(each.split(':')[-1]))
    #         entities_mean.append(np.mean(tmp))
    #     else:
    #         entities_mean.append(0)
    # X_data['entities_mean'] = entities_mean

    # entities_max = []
    # for each_entity in X_data['entities']:
    #     if each_entity != 'null;':
    #         tmp = []
    #         for each in each_entity.strip(';').split(';'):
    #             tmp.append(float(each.split(':')[-1]))
    #         entities_max.append(max(tmp))
    #     else:
    #         entities_max.append(0)
    # X_data['entities_max'] = entities_max

    entities_min = []
    for each_entity in X_data['entities']:
        if each_entity != 'null;':
            tmp = []
            for each in each_entity.strip(';').split(';'):
                tmp.append(float(each.split(':')[-1]))
            entities_min.append(max(tmp))
        else:
            entities_min.append(0)
    X_data['entities_min'] = entities_min

    # entities_std = []
    # for each_entity in X_data['entities']:
    #     if each_entity != 'null;':
    #         tmp = []
    #         for each in each_entity.strip(';').split(';'):
    #             tmp.append(float(each.split(':')[-1]))
    #         entities_std.append(np.std(tmp))
    #     else:
    #         entities_std.append(0)
    # X_data['entities_std'] = entities_std

    entities_len = []
    for each_entity in X_data['entities']:
        if each_entity != 'null;':
            entities_len.append(len(each_entity.strip(';').split(';')))
        else:
            entities_len.append(0)
    X_data['entities_len'] = entities_len
    logger.info('Create feature: entities_min, entities_len.')

    # mentions
    mentions_count_dict = {}
    for each_mention in X_data['mentions']:
        if each_mention != 'null;':
            for each in each_mention.strip().split(' '):
                mentions_count_dict[each] = mentions_count_dict.get(each, 0) + 1

    mentions_len = []
    for each_mention in X_data['mentions']:
        if each_mention != 'null;':
            mentions_len.append(len(each_mention.split(' ')))
        else:
            mentions_len.append(0)
    X_data['mentions_len'] = mentions_len

    mentions_value = []
    for each_mention in X_data['mentions']:
        if each_mention != 'null;':
            tmp = 0
            for each in each_mention.strip().split(' '):
                tmp += mentions_count_dict[each]
            mentions_value.append(tmp)
        else:
            mentions_value.append(0)
    X_data['mentions_value'] = mentions_value
    logger.info('Create feature: mentions_len, mentions_value.')

    # hashtags
    hashtags_count_dict = {}
    for each_hashtag in X_data['hashtags']:
        if each_hashtag != 'null;':
            for each in each_hashtag.strip().split(' '):
                hashtags_count_dict[each] = hashtags_count_dict.get(each, 0) + 1

    hashtags_len = []
    for each_hashtag in X_data['hashtags']:
        if each_hashtag != 'null;':
            hashtags_len.append(len(each_hashtag.split(' ')))
        else:
            hashtags_len.append(0)
    X_data['hashtags_len'] = hashtags_len

    hashtags_value = []
    for each_hashtag in X_data['hashtags']:
        if each_hashtag != 'null;':
            tmp = 0
            for each in each_hashtag.strip().split(' '):
                tmp += hashtags_count_dict[each]
            hashtags_value.append(tmp)
        else:
            hashtags_value.append(0)
    X_data['hashtags_value'] = hashtags_value
    logger.info('Create feature: hashtags_len, hashtags_value.')

    # urls
    X_data['urls_count'] = X_data['urls'].map(lambda x: 1 if x != 'null;' else 0)
    logger.info('Create feature: urls_count.')

    used_feature = ['timestamp', '#followers', '#friends', '#favorites', 'day', 'hour', 'sentiment_left',
                    'sentiment_right', 'sentiment_sum', 'sentiment_diff', 'entities_min', 'entities_len',
                    'mentions_len', 'mentions_value', 'hashtags_len', 'hashtags_value', 'urls_count',
                    'usercount', 'username_lbl', 'entities_lbl', 'mentions_lbl', 'hashtags_lbl', 'urls_lbl',
                    'inner_favorite']

    train = pd.concat([X_data[:num_of_train], pd.DataFrame(np.log(1.0 + y_train), columns=['solution'])], axis=1)
    X_valid = X_data[num_of_train:num_of_train+num_of_valid]
    X_test = X_data[num_of_train+num_of_valid:]

    train[used_feature + ['solution']].to_csv(os.path.join(INPUT_PATH, 'train.csv'), index=False)
    X_valid[used_feature].to_csv(os.path.join(INPUT_PATH, 'X_valid.csv'), index=False)
    X_test[used_feature].to_csv(os.path.join(INPUT_PATH, 'X_test.csv'), index=False)
    logger.info('Preprcessed Data saved in train.csv, X_valid.csv and X_test.csv')