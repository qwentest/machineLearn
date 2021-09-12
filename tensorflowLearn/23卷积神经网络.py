# coding: utf-8 
# @时间   : 2021/9/10 4:10 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 23卷积神经网络.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 前向传播求y，反向传播求w,梯度，偏导的向量。
    # 反射传播，梯度下降 + 链式法则。
    # 一次前向传播 + 一次反向传播 = 更新w一次。
    # 卷积核的数量设定，一般用2的n次方来进行设定。
    # layer数，跟输出的分类数相等。参考已有类型的分辨率和分类数。卷积，卷积、池化，卷积，卷积、池化。
    # 分类问题用交叉熵，回归问题用均方差。
    layer = layers.Conv2D(4, kernel_size=(3, 3), strides=1, padding='SAME')
    # epoch = N / batch 全局梯度下降：计算复杂、落入局部最优梯度消失。
    # 约定大于配置 错误是一种常态。
    # 一切皆为张量；所有的计算都可以表达成由tensor组成的flow;
