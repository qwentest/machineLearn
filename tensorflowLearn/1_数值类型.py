# coding: utf-8 
# @时间   : 2021/9/8 9:23 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 1_数值类型.py
"""
数值类型：
1. 标量(scalar)，单个实数，shape为[],dim为0;
2. 向量(vector)，n个实数的有序集合，如[1. 2. 3.]，shape为[n]，dim为1;
3. 矩阵(matrix)，n行m列的实数的序集合,[[1,2],[2,3]],shape为[n,m],dim为2;
4. 张量(tensor)，所有dim > 2的数组统称为张量;

在tensorflow中这几种数据类型统称炎张量。
"""
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    a = 1.2

    aa = tf.constant(1.2)  # 创建标量
    bb = tf.constant([1, 2, 3])  # 创建向量
    cc = tf.constant([[1, 2], [2, 3]])  # 创建矩阵
    ee = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 创建张量
    ff = tf.constant([[True, False]])  # 布尔类型的张量
    print(aa, bb, cc, ee, ff, sep='\n-----------------------\n')
    ################################
    # 类型转换
    # 数据类型之间的转换
    tf.cast(cc, tf.float32)
    ################################
    # 拥有梯度变化
    # 转换为tf.variable()类型，将普通张量转换为待优化张量。专门用来支持梯度信息的记录的数据类型。
    # tf.variable()，指需要计算梯度的张量，也就是说可能值会更新。
    # 即如果是θ，定义为Variable类型，如果是X，则为constant
    gg = tf.Variable(ee)
    print('\n-----------------------\n')
    print(gg.name, gg.trainable)  # trainable是否需要被优化
    hh = tf.Variable([[1, 2], [3, 4]])
    print('\n-----------------------\n')
    print(hh)
    ################################
    # 从列表和numpy来转换成张量
    print('\n-----------------------\n')
    ii = tf.convert_to_tensor(np.array([[1, 2], [2, 3]]))
    print(ii)
    ################################
    # 创建全为0或者1的张量，如偏置项的初始化。
    print('\n-----------------------\n')
    print(tf.zeros((2, 3)), tf.ones((2, 3)))
    ################################
    # 创建满足正态分布的张量，比如卷积神经网络中的，初始卷积核的张量。
    print('\n-----------------------\n')
    print(tf.random.normal((3, 3)))
    print('\n-----------------------\n')
    print(tf.random.normal((3, 3), mean=1, stddev=2))  # 创建均值为1，标准差为2的正态分布
    # 指定区间分布均匀分布的张量
    print('\n-----------------------\n')
    print(tf.random.uniform((3, 3)))
    ################################
    print('\n-----------------------\n')
    print(tf.range(1, 10, 0.2))
