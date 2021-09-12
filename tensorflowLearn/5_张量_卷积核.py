# coding: utf-8 
# @时间   : 2021/9/8 11:35 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 5_张量_卷积核.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # [b,h,w,c] ,b样本的数量，h,w,高和宽,c通道数

    x = tf.random.normal([4, 32, 32, 3])
    # 创建卷积神经网络
    # filters 卷积核个数的变化,这里定义的是16个;
    # kernel_size 参数 表示卷积核的大小;
    # strides  步长同样会影响输出的前两个维度;卷积计算时，可以隔几个来进行。
    # padding 是否对周围进行填充，“same” 即使通过kernel_size 缩小了维度，但是四周会填充 0，
    # 保持原先的维度；“valid”表示存储不为0的有效信息。
    # 卷积层的描述 https://blog.csdn.net/godot06/article/details/105054657
    layer = layers.Conv2D(16, kernel_size=(2, 2))
    out = layer(x)  # 前向计算
    print(out.shape)
