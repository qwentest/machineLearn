# coding: utf-8 
# @时间   : 2021/9/9 9:57 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 12_张量的比较.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    out = tf.random.normal([100, 10])
    out = tf.nn.softmax(out, axis=1)
    pred = tf.argmax(out, axis=1)
    y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
    result = tf.equal(pred, y)  # 真实值与预测值之间的比较
    # tf.math.greater a>b
    # tf.math.less a<b
    # tf.math.greater_equal a>=b
    # tf.math.less_equal a<=b
    # tf.math.not_equal a!=b
    # tf.math.is_nan a = nan
    out = tf.cast(result, dtype=tf.float32)
    right = tf.reduce_sum(out)
    print(right)
