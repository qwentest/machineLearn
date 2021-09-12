# coding: utf-8 
# @时间   : 2021/9/9 9:38 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 11_张量数据统计.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # l1 范数，定义为向量x所有元素绝对值之和
    # l2 范数，定义为向量x所有元素的平方和，再开根号
    # ∞  范数，定义为向量x所有元素的绝对值的最大数
    x = tf.ones([2, 2])
    # ord指定为1和2时，指l1、l2范数
    print(tf.norm(x, ord=1))
    print(tf.norm(x, ord=2))
    print(tf.norm(x, ord=np.inf))

    x = tf.random.normal([4, 10])
    print(tf.reduce_max(x, axis=1))  # 最大值
    print(tf.reduce_min(x, axis=1))  # 最小值
    print(tf.reduce_mean(x, axis=1))  # 平均值
    print(tf.reduce_sum(x, axis=-1))  # 求和，-1代表最后一个维度。

    # 当不指定axis时，会返回全局最大值、最小值、平均值
    print(tf.reduce_mean(x))

    out = tf.random.normal([2, 10])
    # Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显
    out = tf.nn.softmax(out, axis=1)
    result = tf.argmax(out, axis=1)  # 选取概率最大的值
    print(result)
