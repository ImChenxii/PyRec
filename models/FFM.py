'''
@author: chenxi
@file: FFM.py
@time: 2019/9/10 19:07
@desc: Field-aware Factorization Machine
'''

import tensorflow as tf
from Utils.Data4PyRec import Data4PyRec
from tensorflow.losses import logloss

class FFM():
    def __init__(self, data, label, feature_field, embedding_size=8, lr_reg_l1=0, lr_reg_l2=0, fm_reg_l1=0, fm_reg_l2=0, loss="logloss", metric="logloss", opt="adam", learning_rate=0.1, epochs=10, batch_size=256, verbos=1, random_seed=2018):
        # 数据参数
        self.data = data # 训练特征集
        self.label = label # 训练标签集
        self.feature_num = data.shape[1] # 特征的个数
        self.feature_field = feature_field # feature_field是一个每个feature所属的field列表
        self.field_num = len(set(feature_field)) # 特征所属的field数量

        # 算法特性参数
        self.embedding_size = embedding_size

        # 算法训练参数
        self.lr_reg_l1 = lr_reg_l1  # LR部分L1正则化系数
        self.lr_reg_l2 = lr_reg_l2  # LR部分L2正则化系数
        self.fm_reg_l1 = fm_reg_l1  # FM部分L1正则化系数
        self.fm_reg_l2 = fm_reg_l2  # FM部分L2正则化系数
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

    def init_graph(self):
        # 构建tf计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)  # 设置随机种子大小
            # 设置输入输出
            self.X = tf.placeholder('float', [None, self.feature_num], name='X')  # 输入矩阵维度: m*n,m为数据量大小,n是特征个数
            self.Y = tf.placeholder('float', [None, 1], name='Y')  # 目标值维度: m*1

            # part1: LR部分
            self.w_0 = tf.Variable(tf.truncated_normal([1]), name="w_0") # w_0
            self.w = tf.Variable(tf.truncated_normal([self.feature_num]), name="w") # w
            self.lr_output = tf.add(self.w_0, tf.reduce_sum(tf.multiply(self.w, self.X), axis=1, keep_dims=True), name="LR_part") # W0+WX

            # part2: FFM特征和filed及其他特征交叉项部分
            self.V = tf.Variable(tf.truncated_normal([self.feature_num, self.field_num, self.embedding_size])) # 大小为kfn的嵌入矩阵
            self.ffm_output = tf.Variable(0.0, tf.float32)
            # 外层遍历所有的特征
            for feature_index1 in range(self.feature_num):
                field_index_1 = int(self.feature_field[feature_index1])
                # 内层遍历feature_index1之后的特征
                for feature_index2 in range(feature_index1+1, self.feature_num):
                    field_index_2 = int(self.feature_field[feature_index2])
                    # 左向量，对应V_{i,fj}
                    # VectorSize对应每个隐向量的长度
                    vectorLeft = tf.convert_to_tensor([[feature_index1, feature_index1, i] for i in range(self.embedding_size)])
                    # 在多维上进行索引去除对应的值
                    weightLeft = tf.gather_nd(self.V, vectorLeft)
                    weightLeftAfterCut = tf.squeeze(weightLeft)

                    # 右向量，对应V_{j,fi}
                    vectorRight = tf.convert_to_tensor([[feature_index2, feature_index1, i] for i in range(self.embedding_size)])
                    weightRight = tf.gather_nd(self.V, vectorRight)
                    weightRightAfterCut = tf.squeeze(weightRight)

                    tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut, weightRightAfterCut))

                    indices2 = [feature_index1]
                    indices3 = [feature_index2]

                    x_i = tf.squeeze(tf.gather_nd(self.data, indices2))
                    x_j = tf.squeeze(tf.gather_nd(self.data, indices3))

                    product = tf.reduce_sum(tf.multiply(x_i, x_j))

                    secondItemVal = tf.multiply(tempValue, product)

                    tf.assign(self.ffm_output, tf.add(self.V, secondItemVal))

            self.output = tf.add(self.lr_output, self.ffm_output)

            # 定义目标损失
            self.obj_loss = logloss(self.Y, tf.nn.sigmoid(self.output))

            # 定义正则化损失
            self.lr_l1 = tf.constant(self.lr_reg_l1, name="lr_l1")
            self.lr_l2 = tf.constant(self.lr_reg_l2, name="lr_l2")
            self.fm_l1 = tf.constant(self.fm_reg_l1, name="fm_l1")
            self.fm_l2 = tf.constant(self.fm_reg_l2, name="fm_l2")

            self.l1_norm = tf.reduce_sum(tf.add(tf.multiply(self.lr_l1, tf.abs(self.w)), tf.multiply(self.fm_l1, tf.abs(self.V))))  # L1正则化损失函数
            self.l2_norm = tf.reduce_sum(tf.add(tf.multiply(self.lr_l2, tf.pow(self.w, 2)), tf.multiply(self.fm_l2, tf.pow(self.V, 2))))  # L2正则化损失函数
            self.norm_loss = tf.add(self.l1_norm, self.l2_norm)  # 合并正则化损失

            # 整体损失函数
            self.loss_fun = tf.add(self.obj_loss, self.norm_loss)

            # 选择优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_fun)

            # 初始化
            self.saver = tf.train.Saver()  # 模型保存器
            init = tf.global_variables_initializer()  # 初始化变量
            self.sess = tf.Session()  # 初始化tf会话
            self.sess.run(init)  # 执行初始化变量

    # 在一个batch上训练数据
    def train_in_one_batch(self, batch_X, batch_Y):
        feed_dict = {
            self.X: batch_X,
            self.Y: batch_Y,
        }
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

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




