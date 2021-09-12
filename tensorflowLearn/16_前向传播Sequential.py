# coding: utf-8 
# @时间   : 2021/9/9 2:59 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 16_前向传播Sequential.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential

if __name__ == '__main__':
    x = tf.random.normal([4, 28 * 28])
    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(10, activation=None)
    ])
    out = model(x)
    print(out)

