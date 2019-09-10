import tensorflow as tf
from Utils.Data4PyRec import Data4PyRec
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error

class FM():
    def __init__(self, data, label, embedding_size=8, lr_reg_l1=0, lr_reg_l2=0, fm_reg_l1=0, fm_reg_l2=0, loss="logloss", metric="logloss", opt="adam", learning_rate=0.1, epochs=10, batch_size=256, verbos=1, random_seed=2018):
        # 数据参数
        self.data = data
        self.label = label
        self.feature_num = data.shape[1] # 特征个数

        # 算法的超参数
        self.embedding_size = embedding_size # 嵌入矩阵V的大小
        self.lr_reg_l1 = lr_reg_l1 # LR部分L1正则化系数
        self.lr_reg_l2 = lr_reg_l2 # LR部分L2正则化系数
        self.fm_reg_l1 = fm_reg_l1 # FM部分L1正则化系数
        self.fm_reg_l2 = fm_reg_l2 # FM部分L2正则化系数

        # 训练参数
        self.loss = loss # 损失函数类型
        self.metric = metric # 评价函数类型
        self.opt = opt # 优化器类型
        self.learning_rate = learning_rate # 学习率大小
        self.epochs = epochs # 训练迭代次数
        self.batch_size = batch_size # 一个batch数据大小
        self.verbos = verbos # 打印输出间隔,默认是1个batch一次打印,小于1为不打印输出

        # 其他参数
        self.random_seed = random_seed # 随机数种子大小

    def __del__(self):
        print("FM task over")
        self.sess.close() # 停止会话,防止内存泄露

    def init_graph(self):
        '''
        Function: build FM tensorflow calculation graph
        Returns:
        '''
        # 构建tf计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed) # 设置随机种子大小
            # 设置输入输出
            self.X = tf.placeholder('float', [None, self.feature_num], name='X') # 输入矩阵维度: m*n,m为数据量大小,n是特征个数
            self.Y = tf.placeholder('float', [None, 1], name='Y') # 目标值维度: m*1

            # part1: 逻辑回归线性部分
            # 逻辑回归变量
            self.w_0 = tf.Variable(tf.zeros([1])) # w0变量,可看做1*1大小
            self.w = tf.Variable(tf.zeros[self.feature_num]) # w矩阵，可看做1*n大小
            # 逻辑回归计算式
            self.lr_part = tf.add(self.w_0, tf.reduce_sum(tf.multiply(self.w, self.X), axis=1, keep_dims=True)) # 最终输出维度为: m*1

            # part2: FM交叉项部分
            # V:嵌入矩阵
            self.V = tf.Variable(tf.random_normal([self.embedding_size, self.feature_num], mean=0, stddev=0.01)) # 维度为: k*n
            self.fm_cross_part = 0.5 * tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(self.X, tf.transpose(self.V)), 2)), tf.matmul(tf.pow(self.X, 2), tf.transpose(tf.pow(self.V, 2))), axis=1, keep_dims=True) # 维度为: m*1

            # part3: 合并LR和交叉项的输出
            self.output = tf.add(self.lr_part, self.fm_cross_part) # 维度为: m*1


            # 定义目标损失
            self.obj_loss = tf.losses.log_loss(self.Y, tf.nn.sigmoid(self.output)) # 默认使用对数损失函数
            if self.loss == "auc":
                self.obj_loss = tf.metrics.auc(self.Y, tf.nn.sigmoid(self.output)) # AUC
            elif self.loss == "mse":
                self.obj_loss = tf.losses.mean_squared_error(self.Y, tf.nn.sigmoid(self.output)) # 均方损失函数
            elif self.loss == "mae":
                self.obj_loss = tf.losses.absolute_difference(self.Y, tf.nn.sigmoid(self.output)) # 绝对损失函数


            # 定义正则化损失
            self.lr_l1 = tf.constant(self.lr_reg_l1, name="lr_l1")
            self.lr_l2 = tf.constant(self.lr_reg_l2, name="lr_l2")
            self.fm_l1 = tf.constant(self.fm_reg_l1, name="fm_l1")
            self.fm_l2 = tf.constant(self.fm_reg_l2, name="fm_l2")

            self.l1_norm = tf.reduce_sum(tf.add(tf.multiply(self.lr_l1, tf.abs(self.w)), tf.multiply(self.fm_l1, tf.abs(self.V)))) # L1正则化损失函数
            self.l2_norm = tf.reduce_sum(tf.add(tf.multiply(self.lr_l2, tf.pow(self.w, 2)), tf.multiply(self.fm_l2, tf.pow(self.V, 2)))) # L2正则化损失函数
            self.norm_loss = tf.add(self.l1_norm, self.l2_norm) # 合并正则化损失

            loss = tf.add(self.obj_loss, self.norm_loss)

            # 选择优化器
            if self.opt == "GD": # 梯度下降优化器
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
            elif self.opt == "Adam": # Adam优化器
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
            elif self.opt == "AdaGrad": # AdaGrad优化器
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(loss)
            elif self.opt == "Mometum": # 动量下降优化器
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(loss)

            # 初始化
            self.saver = tf.train.Saver() # 模型保存器
            init = tf.global_variables_initializer() # 初始化变量
            self.sess = tf.Session() # 初始化tf会话
            self.sess.run(init) # 执行初始化变量

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
        pyRecData = Data4PyRec(self.data, self.label, batch_size=self.batch_size, is_shuffle=True, random_seed=self.random_seed)
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
                    if (pyRecData.get_seq() % self.verbos == 0):
                        print("current " + str(self.loss) + " is : " + str(cur_loss) + "!")
            pyRecData.reset()

    # 使用指定的评价函数进行评估,评价函数可以和之前的目标函数不相同
    def evaluate(self, eva_X, eva_Y):
        pred = self.predict(eva_X, eva_Y)
        # 默认是logloss
        metric_fun = log_loss
        if self.metric == "auc": # roc_auc
            metric_fun = roc_auc_score
        elif self.metric == "mae": # mae
            metric_fun = mean_absolute_error
        elif self.metric == "mse": # mse
            metric_fun = mean_squared_error

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
            if pyRecData_for_pre.get_seq() == 1:
                predict = predict_part
            else:
                predict = tf.concat(0, [predict, predict_part])
        return predict

    # 保存模型到指定的path路径
    def save_model(self, path):
        self.saver.save(self.sess, path)



















