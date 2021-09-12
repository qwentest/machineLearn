# coding: utf-8 
# @时间   : 2021/9/8 4:56 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 8_数学运算.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # 支持 + - * / **  tf.power() tf.square() tf.sqrt() tf.exp(3.) tf.math.log(10.)
    a = tf.range(5)
    b = tf.constant(2)

    # 矩阵相乘，A的倒数第一个维度与B的倒数第二个维度长度（行）必须相等。
    # 当张量的维度大于2时，TensorFlow只会选择A和B最后两个维度进行矩阵相乘。
    a = tf.random.normal([4, 3, 28, 32])
    b = tf.random.normal([4, 3, 32, 2])
    print(a @ b)

    a = tf.random.normal([4, 28, 32])
    b = tf.random.normal([32, 16])
    print(tf.matmul(a, b))  # 向量相乘。维度不同时，会自动广播。


