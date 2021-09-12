# coding: utf-8 
# @时间   : 2021/9/9 3:46 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 17_交叉熵.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential

if __name__ == '__main__':
    """
    1.o ∈ R^d，          实数空间，是一个线型问题，所以在输出层可以不加激活函数。
    2.o ∈ 0 or 1，       二分类问题，输出层要加激活函数。
    3.o ∈ [0,1] 其和为1， 多分类问题，需要使用softmax()函数，将输出值映射到[0,1]之间。
    4.o ∈ [-1,1]，       使用tanh激活函数
    """
    z = tf.random.normal([2, 10])
    y_onehot = tf.constant([1, 3])
    y_onehot = tf.one_hot(y_onehot, depth=10)
    # keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=False)
    # from_logits=False,表示经过soft_max进行输出
    # from_logits=True,表示未经过soft_max进行输出
    loss = keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
    loss = tf.reduce_mean(loss)  # 计算交叉熵损失
    print(loss)
    # 另一种方式
    #
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss2 = criteon(y_onehot, z)  # 计算损失
    print(loss2)

    # 最小化交叉熵损失函数的过程也是最大化正确类别的预测概率的过程。
