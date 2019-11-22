'''
@author: chenxi
@file: DeepFM.py
@time: 2019/9/11 14:57
@desc:
'''

import tensorflow as tf
import numpy as np
from Utils.Data4PyRec import Data4PyRec
from Utils.optimizer_select import optimizer_select
from Utils.metric_select import metric_select

class DeepFM():
    def __init__(self, data, label, feature_field, embedding_size=8, deep_layers=[16, 16], reg_l1=0, reg_l2=0, loss="logloss", metric="logloss", opt="Adam", learning_rate=0.1, epochs=10, batch_size=256, verbos=1, random_seed=2018):
        # 数据参数
        self.data = data # 训练特征集
        self.label = label # 训练标签集
        self.feature_num = data.shape[1] # 特征的个数
        self.feature_field = feature_field # feature_field是一个每个feature所属的field列表
        self.field_num = len(set(feature_field)) # 特征所属的field数量

        # 算法特性参数
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers

        # 算法训练参数
        self.reg_l1 = reg_l1 # L1正则化系数
        self.reg_l2 = reg_l2 # L2正则化系数
        self.loss = loss  # 损失函数类型
        self.metric = metric  # 评价函数类型
        self.opt = opt  # 优化器类型
        self.learning_rate = learning_rate  # 学习率大小
        self.epochs = epochs  # 训练迭代次数
        self.batch_size = batch_size  # 一个batch数据大小
        self.verbos = verbos  # 打印输出间隔,默认是1个batch一次打印,小于1为不打印输出

        # 其他参数
        self.random_seed = random_seed

        # 初始化计算图
        self.init_graph()

    def __del__(self):
        print("FFM task over")
        self.sess.close()  # 对象销毁时,停止会话,防止内存泄露

    @property
    def init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            # 数据输入部分
            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index") # 维度为m*f,每个值是每个field第几个取值
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value") # 维度为m*f,每个值是每个field对应的取值

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label") # 真实标签值
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm") # fm后经过的dropout keep系数
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep") # dnn层经过后的dropout keep系数
            self.train_phase = tf.placeholder(tf.bool, name="train_phase") # 决定数据流向的布尔值

            # 模型计算图部分
            # part1: FM
            feature_embedding = tf.Variable(tf.random_normal([self.feature_num, self.embedding_size], 0.0, 1.0), name="feature_embedding") # Embedding层
            self.embeddings = tf.nn.embedding_lookup(feature_embedding, self.feature_index) # 根据feature_index得到的embedding嵌入矩阵向量
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_num, 1]) # 将输入矩阵维度转换为和嵌入矩阵向量能够相乘的格式

            # FM的线性部分(FM的权重是和Deep部分的嵌入共享的)
            feature_bias = tf.Variable(tf.random_normal([self.feature_num, 1], 0.0, 1.0), "feature_bias") # 线性部分权重,即wx中的w
            self.fm_lr = tf.nn.embedding_lookup(feature_bias, self.feature_index) # 取出指定的特征的w
            self.fm_lr = tf.reduce_sum(tf.multiply(self.fm_lr, feature_value), 2) # 计算wx
            self.fm_lr = tf.nn.dropout(self.fm_lr, 0.5) # 防止过拟合的dropout

            # FM的交叉项部分(同FM中的做法,即先在公式层面化简,再求解,可减小求解复杂度)
            self.fm_cross_1 = tf.square(tf.reduce_sum(self.embeddings, 1)) # 交叉部分1,先求和再平方,结果m*k大小
            self.fm_cross_2 = tf.reduce_sum(tf.square(self.embeddings), 1) # 交叉部分2,先平方再求和.结果维度field大小
            self.fm_cross = 0.5 * tf.subtract(self.fm_cross_1, self.fm_cross_2) # 两部分相减
            self.fm_cross = tf.nn.dropout(self.fm_cross, 0.5) # 防止过拟合的dropout


            # part2: Deep
            self.deep = tf.reshape(self.embeddings, shape=[-1, self.field_num * self.embedding_size]) # 输入dnn之前的数据维度转换
            self.deep = tf.nn.dropout(self.deep, 0.5) # 防止过拟合的dropout

            input_size = self.embedding_size * self.field_num
            glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
            self.dnn_weights_1 = tf.Variable(np.random.normal(loc=0, scale=glorot))

        pass
        # 训练模型
        def train(self):
            # 加载数据到Data4PyRec类型
            pyRecData = Data4PyRec(self.data, self.label, batch_size=self.batch_size, is_shuffle=True,
                                   random_seed=self.random_seed)
            # 迭代epochs次
            for epoch in range(self.epochs):
                # 数据类型会计算是否还有剩余batch
                while pyRecData.has_next():
                    # 取出下一个batch数据
                    (batch_data, batch_label) = pyRecData.next()
                    # 计算这个batch数据的loss
                    cur_loss = self.train_in_one_batch(batch_data, batch_label)
                    # 打印
                    if self.verbos > 0:
                        if (pyRecData.get_idx() % self.verbos == 0):
                            print("current " + str(self.loss) + " is : " + str(cur_loss) + "!")
                pyRecData.reset()

        # 使用指定的评价函数进行评估,评价函数可以和之前的目标函数不相同
        def evaluate(self, eva_X, eva_Y):
            pred = self.predict(eva_X, eva_Y)
            metric_fun = metric_select.select(self.metric)

            return metric_fun(eva_Y, pred)

        # 对指定的数据集进行预测
        def predict(self, pre_X, pre_Y):
            # 下过程同训练
            pyRecData_for_pre = Data4PyRec(pre_X, pre_Y, batch_size=self.batch_size)
            while pyRecData_for_pre.has_next():
                (batch_data_for_pre, batch_label_for_pre) = pyRecData_for_pre.next()
                feed_dict = {
                    self.X: batch_data_for_pre,
                    self.Y: batch_label_for_pre,
                }
                predict_part = self.sess.run(self.output, feed_dict=feed_dict)
                # 对每个batch的预测结果进行合并
                predict = None
                if pyRecData_for_pre.get_idx() == 1:
                    predict = predict_part
                else:
                    predict = tf.concat(0, [predict, predict_part])
            return predict

        # 保存模型到指定的path路径
        def save_model(self, path):
            self.saver.save(self.sess, path)

