'''
@author: chenxi
@file: optimizer_select.py
@time: 2019/9/11 10:09
@desc:
'''

import tensorflow as tf

class optimizer_select():
    def select(self, opt_type, learning_rate):
        # 默认使用梯度下降优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if opt_type == "Adam":  # Adam优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif opt_type == "AdaGrad":  # AdaGrad优化器
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif opt_type == "Mometum":  # 动量下降优化器
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)

        return optimizer
