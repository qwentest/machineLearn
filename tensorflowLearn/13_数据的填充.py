# coding: utf-8 
# @时间   : 2021/9/9 10:05 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 13_数据的填充.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    a = tf.constant([1, 2, 3, 4, 5, 6])
    b = tf.constant([7, 8, 1, 6])
    # 数据的填充
    # 参数为[left padding, right padding]的嵌套list
    # [[0,0],[2,1],[1,2]] 第1个维度不填充，第2个维度左边填充2个右边填充1个，第3个维度，左边1个右边2个。
    b = tf.pad(b, [[0, 2]])  # 第1个维度不填充，第2个维度填充2个0
    print(b)

    # 模拟图片的填充
    x = tf.random.normal([4, 28, 28, 1])  # 28 * 28 的图像
    tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])  # 32 * 32 的图像


