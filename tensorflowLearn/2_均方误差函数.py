# coding: utf-8 
# @时间   : 2021/9/8 10:47 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 2_均方误差函数.py
import tensorflow as tf

if __name__ == "__main__":
    out = tf.random.uniform([4, 10])
    y = tf.constant([2, 3, 2, 0])
    y = tf.one_hot(y, depth=10)  # 每个向量对应的标签的类型，即多分类
    loss = tf.keras.losses.mse(y, out)  # 均方误差函数，即真实值与预测值之间的差值
    loss = tf.reduce_mean(loss)  # 平均值
    print(loss)
