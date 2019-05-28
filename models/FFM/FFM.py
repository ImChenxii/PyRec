'''
FFM是在FM嵌入的基础上，对其他的field也进行了权重连接，来训练每个特征对于每个field的隐向量
参数大小从原来的每个field只对应自身特征的FM为Kn变为了FFM的Knf
'''
import tensorflow as tf
import numpy as np
import os

all_data_size = 1000


# 模型参数
input_x_size = 20 # 输入维度n
field_size = 2 # field大小f
vector_dimension = 3 # V矩阵维度K

# 训练参数
total_plan_train_steps = 1000
learning_rate = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"

# 三维矩阵，用于创建FFM维度为nfK的交叉矩阵
def createTwoDimensionWeight(input_x_size, field_size, vector_dimension):
    # 正态分布
    weights = tf.truncated_normal([input_x_size, field_size, vector_dimension])
    tf_weights = tf.Variable(weights)
    return tf_weights

# 一维矩阵，用于创建线性部分的模型权重参数
def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    tf_weights = tf.Variable(weights)

    return tf_weights

# 0维矩阵，用于创建线性部分的bias
def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)

    return tf_weights

# 定义前向网络
def inference(input_x, input_x_field, zeroWeights, oneDimWeights, thirdWeights):
    # 公式中的第二项
    secondValue = tf.reduce_sum(tf.multiply(oneDimWeights, input_x, name="secondValue"))
    # 公式中的第一项和第二项之和
    firstSecondValue = tf.add(zeroWeights, secondValue, name="firstSecondValue")
    # 第三项大小
    thirdValue = tf.Variable(0.0, dtype=tf.float32)
    input_shape = input_x_size

    # 第一次遍历遍历所有的特征
    for i in range(input_shape):
        featureIndex1 = i
        fieldIndex1 = int(input_x_field[i])
        # 第二次遍历i特征之后特征，进行组合
        for j in range(i+1, input_shape):
            featureIndex2 = j
            fieldIndex2 = int(input_x_field[j])
            # 左向量，对应V_i fj
            # VectorSize对应每个隐向量的长度
            vectorLeft = tf.convert_to_tensor([[featureIndex1, fieldIndex2, i] for i in range(vector_dimension)])
            # 在多维上进行索引去除对应的值
            weightLeft = tf.gather_nd(thirdWeights, vectorLeft)
            weightLeftAfterCut = tf.squeeze(weightLeft)

            # 右向量，对应V_j fi
            vectorRight = tf.convert_to_tensor([[featureIndex2, fieldIndex1, i] for i in range(vector_dimension)])
            weightRight = tf.gather_nd(thirdWeights, vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut, weightRightAfterCut))

            indices2 = [i]
            indices3 = [j]

            x_i = tf.squeeze(tf.gather_nd(input_x, indices2))
            x_j = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(x_i, x_j))

            secondItemVal = tf.multiply(tempValue, product)

            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

    return tf.add(firstSecondValue, thirdValue)


def gen_data():
    labels = [-1, 1]
    y = [np.random.choice(labels, 1)[0] for _ in range(all_data_size)]
    # 每10个特征属于一个field
    x_field = [i // 10 for i in range(input_x_size)]
    x = np.random.randint(0, 2, size=(all_data_size, input_x_size))
    return x, y, x_field

def train():
    global_step = tf.Variable(0, trainable=False)
    train_x, train_y, trainxField = gen_data()

    input_x = tf.placeholder(tf.float32, [input_x_size])
    input_y = tf.placeholder(tf.float32)

    # 正则化稀疏
    lambda_w = tf.constant(0.001, name="lambda_w")
    lambda_v = tf.constant(0.001, name="lambda_v")

    zeroWeights = createZeroDimensionWeight()
    oneDimWeights = createOneDimensionWeight(input_x_size)
    thirdWeight = createTwoDimensionWeight(input_x_size, field_size, vector_dimension)

    # 前向传播结果
    y_ = inference(input_x, trainxField, zeroWeights, oneDimWeights, thirdWeight)

    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(oneDimWeights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v,  tf.pow(thirdWeight, 2)), axis=[1, 2])
        )
    )

    loss = tf.log(1 + tf.exp(input_y * y_)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(total_plan_train_steps):
            for t in range(total_plan_train_steps):
                input_x_batch = train_x[t]
                input_y_batch = train_y[t]
                predict_loss_, _, steps = sess.run([loss, train_step, global_step],
                                                   feed_dict={input_x: input_x_batch, input_y: input_y_batch})


                print("After {step} training steps, loss on training batch is {predict_loss}".format(step=steps, predict_loss=predict_loss_))


                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()

if __name__ == '__main__':
    train()