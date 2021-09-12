# coding: utf-8 
# @时间   : 2021/9/9 10:34 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 14_数据集加载.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


def preprocess(x, y):
    # 标准化0 - 1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 打平
    x = tf.reshape(x, [-1, 28, 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


if __name__ == "__main__":
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 转换成dataset对象
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # 随机打散
    train_db = train_db.shuffle(10000)
    # 批训练,利用显卡的并行能力来进行处理。
    # train_db = train_db.batch(128)  # 每个批样本的处理数
    # train_db = train_db.map(preprocess)
    # for b in enumerate(train_db):
    #     pass
    # pass
