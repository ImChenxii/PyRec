import tensorflow as tf
import pandas as pd
import numpy as np

train_data = pd.read_csv("./")
train_y = pd.read_csv('./')
test_x = pd.read_csv("./")
test_y = pd.read_csv('./')

# m:训练集大小
# n:维度大小
m, n = train_data.shape
# k: FM的嵌入矩阵大小
k = 10

x = tf.placeholder('float', [None, n])
y = tf.placeholder('float', [None, 1])

# 线性部分参数
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([n]))

# V矩阵(嵌入矩阵)参数
v = tf.variable(tf.random_normal([k, n], mean=0, stddev=0.01))

# 线性模型部分
# 广播形式对应位置元素相乘
linear_part = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))

# FM增加部分 结果维度：m*1
fm_cross = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(x, tf.transpose(v)), 2), tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
    ), axis=1, keep_dims=True
)

y_hat = tf.add(linear_part, fm_cross)

# 定义损失函数
# 定义正则化
lambda_w = tf.constant(0.001, name="lambda_w")
lambda_v = tf.constant(0.001, name="lambda_v")

l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w, tf.pow(w, 2)),
        tf.multiply(lambda_v, tf.pow(v, 2)),
    )
)

# 平方损失
error = tf.reduce_mean(tf.square(y - y_hat))
loss = tf.add(error, l2_norm)

# 定义优化器
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 定义变量初始化
init = tf.global_variables_initializer()

epochs = 10
batch_size = 1000

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

# 训练
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        perm = np.random.permutation(train_data.shape[0])

        for bX, bY in batcher(train_data[perm], train_y[perm], batch_size):
            _, t = sess.run([train_op, loss], feed_dict={
                x: bX.reshape(-1, n), y: bY.reshape(-1, 1)})
            print(t)

        errors = []
        for bX, bY in batcher(test_x, test_y):
            errors.append(sess.run(error, feed_dict={
                x: bX.reshape(-1, n), y: bY.reshape(-1, 1)}))
            print(errors)
        RMSE = np.sqrt(np.array(errors).mean())
        print(RMSE)


