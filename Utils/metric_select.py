'''
@author: chenxi
@file: metric_select.py
@time: 2019/9/11 10:44
@desc:
'''

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, mean_absolute_error

class metric_select():
    def select(self, metric_type):
        metric = log_loss
        if metric_type == "auc": # roc_auc
            metric = roc_auc_score
        elif metric_type == "mae": # mae
            metric = mean_absolute_error
        elif metric_type == "mse": # mse
            metric = mean_squared_error

        return metric
