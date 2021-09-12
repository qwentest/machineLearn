# coding: utf-8 
# @时间   : 2021/9/8 2:14 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 6_索引和切片.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # [b,h,w,c] ,b样本的数量，h,w,高和宽,c通道数

    x = tf.random.normal([4, 32, 32, 3])
    # 按维度挨个进行取值
    print(x[0])  # 第1个样本
    print(x[0][1])  # 第1个样本的第2行
    print(x[0][1][2])  # 第1个样本的第2行的第3列
    print(x[2][1][0][1])  # 第3全术本是的第2行第1列的第2个通道
    print(x[1, 9, 2])  # 第1个样本第10行第2列

    ######################################
    # 切片
    print(x[1:3])  # 第2个和第3个样本
    print(x[0, ::])  # 读取第1个样本
    print(x[0, ...])
    print(x[:, 0:28:2, 0:28:2, :])  # 读取所有的的内容，但隔2行进行取值，维度减半，缩小50%。

    # 没有规则的索引取值，使用tf.gather函数。
    # tf.gather_nd，通过指定每次采样点的多维坐标来实现多样多个点的目的。

