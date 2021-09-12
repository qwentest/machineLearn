# coding: utf-8 
# @时间   : 2021/7/30 8:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 多变量线型回归
# @数据   : ex1data2.txt,增加了一个房间数，所以激活函数就变为了
# h𝝷(χi) =  θ_1 * x_1 + θ_2 * x_2
# @文件   : lineRegression_2.py.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#因为已向量化实现递度下降的过程，所以逻辑没有过多变化。
import machineLearn.lineRegression_1 as line

def feature_normalization(pd_data):
    """
    将数据进行特征归一化处理。
    原因：当进行多个变量的梯度下降时，如果各个变量的取值范围差别很大，
    即这些特征的尺度相差很大，那么梯度下降时收敛速度可能会非常慢，
    因此为了适当的提高收敛的速度，我们可以通过将所有特征通过缩放到统一尺度来加快收敛速度
    均值归一化（Mean normalization）是比较通用的特征缩放的方法：
    即通过让特征x_{i}尽可能接近0，在除以特征取值范围的最大值减去最小值。
    :param pd_data:
    :return:
    """
    data = (pd_data - pd_data.mean()) / pd_data.std()
    return data


if __name__ == "__main__":
    pd_data = line.show_data('./data/ex1data2.txt',
                             saveName='./img/line_2.jpg',columns=['Size', 'Bedrooms', 'Price'],isShow=False)
    pd_data = feature_normalization(pd_data)

    # print(pd_data)
    pd_data.insert(0, 'Ones', 1)
    # 随机选择60%的数据用来做训练集的内容。
    pd_data = pd_data.sample(frac=0.6)
    cols = pd_data.shape[1]  # 列数
    X = pd_data.iloc[:, 0:cols - 1]  # 取前cols-1列，即输入向量X
    y = pd_data.iloc[:, cols - 1:cols]  # 取最后一列，即目标向量y
    # 设置学习率
    alpha = 0.003
    # 设置迭代次数
    epoch = 1100
    # 转换为向量
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # 设置初始theta
    theta = np.matrix([0, 0, 0])
    l = line.LR(X, y)
    theta, cost = l.gradientDescent(theta, alpha, epoch)
    print("预测出来的变量值{},cost={}".format(theta, cost))
    line.verfiy_epoch(epoch,cost,saveName='./img/line_lr_epoch_2.jpg',isShow=False)


