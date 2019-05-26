import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size=8,
                 dropout_fm=[1.0, 1.0], deep_layers=[32, 32],
                 dropout_deep=[0.5, 0.5, 0.5], deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256, learning_rate=0.001, optimizer='adam',
                 batch_norm=0, batch_norm_decay=0.995, verbose=False, random_seed=2019,
                 use_fm=True, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verobse = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        # 初始化时执行初始化图函数
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weight()

            # DeepFM模型

            # FM part
            # 嵌入部分结果
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            # feat_value是原始输入
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            # feat_value经过embedding变为embeddings
            self.embeddings = tf.multiply(self.embeddings, feat_value)
            # first part 线性部分
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

            # second part 交叉部分
            # 交叉部分1，先求和再平方，结果维度field大小
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)
            self.summed_features_emb_square = tf.square(self.summed_features_emb)
            # 交叉部分2，先平方再求和，结果维度field大小
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)

            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            # Deep part
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layers_%d" %i]), self.weights["bias_%d" %i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i+1])

            # Deep and Fm
            # 全连接层
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])


            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # 正则化项
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_reguraizer(self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_reguraizer(self.l2_reg)(self.weights["layer_%d" %i])

            # 优化器
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                        beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            # 初始化
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # 统计训练参数个数
            total_parameters = 0
            for varibales in self.weights.values():
                shape = varibales.get_shape()
                varibales_parameters = 1
                for dim in shape:
                    varibales_parameters *= dim.value
                total_parameters += varibales_parameters
            if self.verobse > 0:
                print("#params: %d" %total_parameters)

    # 设置网络各层的训练权重参数
    def _initialize_weight(self):
        weights = dict()

        # embedding层
        # embedding_size指的是嵌入向量的大小k
        # 初始输入层和embedding层的连接权重个数是特征个数n*embedding向量大小k，每个field对应一个embedding向量
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings'
        )
        # 每个特征嵌入后会得到对应field位置的一个embedding向量，在计算这个embedding向量时有一个线性相加项和一个偏置项
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        # deep层
        num_layer = len(self.deep_layers) # deep层层数
        input_size = self.field_size * self.embedding_size # deep层输入维度大小，因为每个field对应一个embedding向量
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        # 从嵌入层到神经网络第一层的连接权重
        # loc：均值 scale：标准差
        weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)
        # 构建后续deep网络层
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" %i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])), dtype=np.float32
            )
            weights["bias_%d" %i] = tf.Varialbe(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32
            )

        # 最终全连接层的输入大小
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        # FM最终的输出大小是field大小()
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        # 神经网络部分最终输出大小是最后一层的单元数
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (input_size+1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    # 获取batch_size大小数据
    def get_batch(self, X_i, X_v, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return X_i[start: end], X_v[start: end], [[y_] for y_ in y[start: end]]
    # 将数据集随机打散
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)

    # 评估
    def evaluate(self, X_i, X_v, y):
        y_pred = self.predict(X_i, X_v)
        return self.eval_metric(y, y_pred)
    # 预测
    def predcit(self, X_i, X_v):
        dummy_y = [1] * len(X_i)
        batch_index = 0
        X_i_batch, X_v_batch, y_batch = self.get_batch(X_i, X_v, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(X_i_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {
                self.feat_index: X_i_batch,
                self.feat_value: X_v_batch,
                self.label: y_batch,
                self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                self.dropout_keep_deep: [1.0] * len(self.dropout_keep_deep),
                self.train_phase: False
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            X_i_batch, X_v_batch, y_batch = self.get_batch(X_i, X_v, dummy_y, self.batch_size, batch_index)

        return y_pred

    def fit_on_batch(self, X_i, X_v, y):
        feed_dict = {
            self.feat_index: X_i,
            self.feat_value: X_v,
            self.label: y,
            self.dropout_keep_fm: self.dropout_fm,
            self.dropout_keep_deep: self.dropout_keep_deep,
            self.train_phase: True
        }
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def fit(self, X_i_train, X_v_train, y_train, X_i_valid=None, X_v_valid=None,
            y_valid=None, early_stopping=False, refit=False):
        has_valid = X_v_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(X_i_train, X_v_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in (total_batch):
                X_i_batch, X_v_batch, y_batch = self.get_batch(X_i_train, X_v_train, y_train, self.batch_size, i)
                self.fit_on_batch(X_i_batch, X_v_batch, y_batch)

            train_result = self.evaluate(X_i_train, X_v_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(X_i_valid, X_v_valid, y_valid)
                self.valid_result.append(valid_result)

            if self.verobse > 0 and epoch % self.verobse == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          %(epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_terminiation(self.valid_result):
                break

        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            X_i_train = X_i_train + X_i_valid
            X_v_train = X_v_train + X_v_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(X_i_train, X_v_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    X_i_batch, X_v_batch, y_batch = self.get_batch(X_i_train, X_v_train,
                                                                   self.batch_size, i)
                    self.fit_on_batch(X_i_batch, X_v_batch, y_batch)
                train_result = self.evaluate(X_i_train, X_v_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or (self.greater_is_better and train_result > best_train_score) \
                    or ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):
        if (len(valid_result)) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False