# coding: utf-8 
# @时间   : 2021/9/8 11:06 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 3_向量_创建网络层Dense.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    fc = layers.Dense(3)  # 全连接中创建一个网络层，张量w和b存储在某内部，由类自动创建并进行管理。输出节点为3
    fc.build(input_shape=(2, 4))
    print(fc.bias)  # 查看编置向量个数,layers.Dense()创建的是一个Variable的类型。因为其参数是可变的。
    print(fc.kernel)  # 查看权值矩阵
