# coding: utf-8 
# @时间   : 2021/9/8 5:31 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 9_前向传播.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # 第一层输出节点是256
    # 第二层输出节点是128
    # 第三层的输出节点是10
    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
    # 完整代码：https://blog.csdn.net/CSDNXXCQ/article/details/116431031
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))

    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))

    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = tf.reshape(x_train, [-1, 28 * 28], )  # -1,将所有数据转换成多维。因为原图像是 28 * 28，调整为[b,784]

    # x 此时是uint8不能直接于w1进行计算
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
        h1 = tf.nn.relu(h1)  # 另一个激活函数

        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)

        out = h2 @ w3 + b3

        # 将y_train转换成one-hot编码.
        y_onehot = tf.one_hot(y_train, 10)

        # 计算预测网络y值于实际值之间的误差
        loss = tf.square(y_onehot - out)
        loss = tf.reduce_mean(loss)  # 误差平均值
        # tf.GradientTape()，自动求导
        # 参数说明：
        # persistent: 布尔值，用来指定新创建的gradient tape是否是可持续性的。默认是False，意味着只能够调用一次gradient（）函数。
        # watch_accessed_variables: 布尔值，表明这个gradien tap是不是会自动追踪任何能被训练（trainable）的变量。默认是True。
        lr = 0.003

        """
        watch(tensor)
        作用：确保某个tensor被tape追踪
        参数:
        tensor: 一个Tensor或者一个Tensor列表
        """
        # tape.watch([w1, b1, w2, b2, w3, b3])
        """
        根据tape上面的上下文来计算某个或者某些tensor的梯度
        返回:
        一个列表表示各个变量的梯度值，和source中的变量列表一一对应，表明这个变量的梯度。
        """
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 梯度更新，assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

    pass
