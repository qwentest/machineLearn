# coding: utf-8 
# @时间   : 2021/9/8 2:40 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 7_视图改变.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    x = tf.range(96)
    # 改变形状， 需要注意，改变顺序后，每一个维度代表的意义要统一顺序。
    x = tf.reshape(x, [2, 4, 4, 3])
    print(x)
    y = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)
    y = tf.expand_dims(x, axis=2)  # 表示维度后面增加一个维度
    print(y)

    # x = tf.squeeze(x, axis=0)  # 删除第1个维度
    # print(x)

    # [b,h,w,c] - > [b,c,w,h]
    x = tf.random.normal([2, 32, 32, 3])
    # 按某个位置进行转置
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    print(x)
    # 在样本数据上复制出来一份
    x = tf.range(4)
    x = tf.reshape(x, [2, 2])
    # multiples，分别在每个维度上面复制的倍数，1表明不复制，2表明为新长度的2倍，即复制一份。
    # 即在axis = 0 时复制2次，axis = 1上复制1次。
    x = tf.tile(x, multiples=[2, 1])
    print(x)

    # 自动完成维度的增加并复制数据的功能。
    A = tf.random.normal([32, 1])
    B = tf.broadcast_to(A, [2, 32, 32, 3])
    print(B)
