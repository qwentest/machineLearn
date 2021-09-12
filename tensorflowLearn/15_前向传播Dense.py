# coding: utf-8 
# @时间   : 2021/9/9 2:34 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 15_前向传播Dense.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    x = tf.random.normal([4, 28 * 28])
    # 创建一个全链层，指定输出节点数和激活函数
    fc1 = layers.Dense(256, activation=tf.nn.relu)
    fc2 = layers.Dense(128, activation=tf.nn.relu)
    fc3 = layers.Dense(64, activation=tf.nn.relu)
    fc4 = layers.Dense(10, activation=None)
    # 通过fc类实例完成一次全连接层的计算，返回输出张量
    h1 = fc1(x)  # 通过隐藏层1得到输出
    h2 = fc2(h1)  # 通过隐藏层2得到输出
    h3 = fc3(h2)  # 通过隐藏层3得到输出
    h4 = fc4(h3)  # 通过输出层得到网络输出

    # # 获取权值的矩阵
    # w1 = fc.kernel
    # # 获取偏置向量
    # b1 = fc.bias
    # # 返回待优化参数列表
    # train_var = fc.trainable_variables
    pass
