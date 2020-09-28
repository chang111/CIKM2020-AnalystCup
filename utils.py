# coding=utf-8

'''
    工具函数
'''
import numpy as np
from sklearn.metrics import mean_squared_error

def xgb_mse(preds, labels):
    label = labels.get_label()
    score = mean_squared_error(label, preds)
    return 'mse', score

def lgb_mse(preds, labels):
    label = labels.get_label()
    score = mean_squared_error(label, preds)
    return 'mse', score, False

def lgb_int_mse(preds, labels):
    label = labels.get_label()
    score = mean_squared_error(label, np.round(np.maximum(preds, 0)))
    return 'int_mse', score, False