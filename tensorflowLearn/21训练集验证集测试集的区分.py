# coding: utf-8 
# @时间   : 2021/9/10 2:11 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 21训练集验证集测试集的区分.py
"""
验证集与测试集的区分：
1.根据验证集的表现来调整模型的各个超参数的设置，提升模型的泛化能力，但是测试集不能用来反馈模型的调整，否则测试集
与验证集的功能重合，因此在测试集上的性能表现无法代表模型的泛化能力。
超参数包括：
1.根据验证集的性能表现来调整学习率、权重衰减系数、训练次数等。
2.根据验证集的性能表现来重新调整网络拓扑结构。
3.根据验证集的性能表现来判断是否过拟合和欠拟合。

训练准确率高，验证低，过拟合；
1.降低网络层数，降低网络参数量，添加正则化，添加假设约束等。
2.增加dropout层，dropout层随机断开神经网络的连接，减少每次训练的参数量，但是测试的时候又会恢复。
3.增加数据集的规模是解决过拟合的最重要的途径。
  数据增强：在维持样本标签不变的条件下，根据先验知识改变样本的特征，使得新产生的数据样本更符合或者近似真实数据的分布。


训练准确率低，验证也低，欠拟合；
2.增加网络层数，增加网络参数量。

什么时候是最佳状态：
1.训练集和验证集随着训练的次数，两者误差会有一个交叉点，此时最佳，也就是肘点法则。
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential, regularizers
import matplotlib.pyplot as plt

if __name__ == "__main__":
    w1 = tf.random.normal([4, 3])
    w2 = tf.random.normal([4, 2])
    # l1范数，定义为所有张量绝对值之和
    loss_reg1 = tf.reduce_sum(tf.math.abs(w1)) + tf.reduce_sum(tf.math.abs(w1))
    # l2范数，定义为所有张量的平方之和
    loss_reg2 = tf.reduce_sum(tf.math.sqrt(w1)) + tf.reduce_sum(tf.math.sqrt(w2))

    # model.add(layers.Dropout(rate=0.5))
    # regularizers.l2(c) l2范数,c为正则化的参数。

