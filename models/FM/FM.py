import tensorflow as tf
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_num, embedding_size=8, lr_reg_l1=0, lr_reg_l2=0, fm_reg_l1=0, fm_reg_l2=0, loss="logloss", opt="adam", learning_rate=0.1, epochs=10, batch_size=256, random_seed=2018):

        # 数据参数
        self.feature_num = feature_num # 特征个数

        # 算法的超参数
        self.embedding_size = embedding_size # 嵌入矩阵V的大小
        self.lr_reg_l1 = lr_reg_l1 # LR部分L1正则化系数
        self.lr_reg_l2 = lr_reg_l2 # LR部分L2正则化系数
        self.fm_reg_l1 = fm_reg_l1 # FM部分L1正则化系数
        self.fm_reg_l2 = fm_reg_l2 # FM部分L2正则化系数



        # 训练参数
        self.loss = loss # 损失函数类型
        self.opt = opt # 优化器类型
        self.learning_rate = learning_rate # 学习率大小
        self.epochs = epochs # 训练迭代次数
        self.batch_size = batch_size # 一个batch数据大小

        # 其他参数
        self.random_seed = random_seed # 随机数种子大小



    def init_graph(self):
        '''
        Function: build FM tensorflow calculation graph
        Returns:
        '''
        # 构建tf计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed) # 设置正则化参数
            # 设置输入输出
            train_x = tf.placeholder('float', [None, self.feature_num], name='X') # 输入矩阵维度: m*n,m为数据量大小,n是特征个数
            label = tf.placeholder('float', [None, 1], name='label') # 目标值维度: m*1

            # part1: 逻辑回归线性部分
            # 逻辑回归变量
            w_0 = tf.Variable(tf.zeros([1])) # w0变量,可看做1*1大小
            w = tf.Variable(tf.zeros[self.feature_num]) # w矩阵，可看做1*n大小
            # 逻辑回归计算式
            lr_part = tf.add(w_0, tf.reduce_sum(tf.multiply(w, train_x), axis=1, keep_dims=True)) # 最终输出维度为: m*1

            # part2: FM交叉项部分
            # V:嵌入矩阵
            V = tf.Variable(tf.random_normal([self.embedding_size, self.feature_num], mean=0, stddev=0.01)) # 维度为: k*n
            fm_cross_part = 0.5 * tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(train_x, tf.transpose(V)), 2)), tf.matmul(tf.pow(train_x, 2), tf.transpose(tf.pow(V, 2))), axis=1, keep_dims=True) # 维度为: m*1

            # part3: 合并LR和交叉项的输出
            output = tf.add(lr_part, fm_cross_part) # 维度为: m*1


            # 定义目标损失
            obj_loss = tf.losses.log_loss(label, tf.nn.sigmoid(output)) # 默认使用对数损失函数
            if self.loss == "auc":
                obj_loss = tf.metrics.auc(label, tf.nn.sigmoid(output)) # AUC
            elif self.loss == "mse":
                obj_loss = tf.losses.mean_squared_error(label, tf.nn.sigmoid(output)) # 均方损失函数
            elif self.loss == "mae":
                obj_loss = tf.losses.absolute_difference(label, tf.nn.sigmoid(output)) # 绝对损失函数


            # 定义正则化损失
            lr_l1 = tf.constant(self.lr_reg_l1, name="lr_l1")
            lr_l2 = tf.constant(self.lr_reg_l2, name="lr_l2")
            fm_l1 = tf.constant(self.fm_reg_l1, name="fm_l1")
            fm_l2 = tf.constant(self.fm_reg_l2, name="fm_l2")

            l1_norm = tf.reduce_sum(tf.add(tf.multiply(lr_l1, tf.abs(w)), tf.multiply(fm_l1, tf.abs(V)))) # L1正则化损失函数
            l2_norm = tf.reduce_sum(tf.add(tf.multiply(lr_l2, tf.pow(w, 2)), tf.multiply(fm_l2, tf.pow(V, 2)))) # L2正则化损失函数
            norm_loss = tf.add(l1_norm, l2_norm) # 合并正则化损失

            loss = tf.add(obj_loss, norm_loss)

            # 选择优化器
            if self.opt == "GD": # 梯度下降优化器
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
            elif self.opt == "Adam": # Adam优化器
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
            elif self.opt == "AdaGrad": # AdaGrad优化器
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(loss)
            elif self.opt == "Mometum": # 动量下降优化器
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(loss)

            # 初始化saver
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.sess.close()

















