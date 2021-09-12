# coding: utf-8 
# @时间   : 2021/9/10 3:03 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 22图片处理.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
import matplotlib.pyplot as plt


def preprocess(x):
    x = tf.io.read_file(x)  # 读
    x = tf.image.decode_jpeg(x, channels=3)  # 转成RGBA
    x = tf.image.resize(x, [244, 244])  # 重置大小
    x = tf.image.rot90(x, 2)  # 旋转180度
    x = tf.image.random_flip_left_right(x)  # 随机水平翻转
    x = tf.image.random_flip_up_down(x)  # 随机竖直翻转
    x = tf.image.random_crop(x, [244, 244, 3])  # 随机裁剪到合适尺寸
    return x


if __name__ == "__main__":
    pass
