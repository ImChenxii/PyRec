'''
@author: chenxi
@file: loss_select.py
@time: 2019/9/11 10:22
@desc:
'''
from tensorflow.losses import mean_squared_error, absolute_difference, log_loss
from tensorflow.metrics import auc

class loss_select():
    def select(self, loss_type, y, y_hat):
        loss = log_loss(y, y_hat)
        if loss_type == "auc":
            loss = auc(y, y_hat)  # AUC
        elif loss_type == "mse":
            loss = mean_squared_error(y, y_hat)  # 均方损失函数
        elif loss_type == "mae":
            loss = absolute_difference(y, y_hat)  # 绝对损失函数

        return loss
