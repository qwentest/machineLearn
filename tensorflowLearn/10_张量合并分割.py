# coding: utf-8 
# @时间   : 2021/9/9 9:20 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 10_张量合并分割.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # 张量的堆叠和拼接。拼接不会产生新的维度，而堆叠会。
    # 合并的惟度长度需要保持一致。
    a = tf.random.normal([4, 35, 8])
    b = tf.random.normal([6, 35, 8])
    joinAB = tf.concat([a, b], axis=0)
    print(joinAB)
    a = tf.random.normal([10, 35, 4])
    b = tf.random.normal([10, 35, 4])
    joinAB_2 = tf.concat([a, b], axis=2)
    print(joinAB_2)

    stackAB = tf.stack([a, b], axis=0)
    print(stackAB)

    # 分割函数
    x = tf.random.normal([10, 35, 8])
    result = tf.split(x, num_or_size_splits=10, axis=0)
    # 数据切割成4份，每份长度分别为[4,2,2,2]
    result1 = tf.split(x, num_or_size_splits=[4, 2, 2, 2], axis=0)
    print(result, result1)