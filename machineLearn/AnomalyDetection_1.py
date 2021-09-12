# coding: utf-8 
# @时间   : 2021/8/12 4:09 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 异常检查的过程
# @文件   : AnomalyDetection_1.py
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def read_data():
    """读取原数据"""
    mat = loadmat('./data/ex8data1.mat')
    # print(mat.keys())
    X = mat['X']
    Xval, yval = mat['Xval'], mat['yval']
    return X, Xval, yval


def plot_data(X):
    """
    展示原有数据。存在部分数据远离中心的主数据内容。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(X[:, 0], X[:, 1], 'bx')


def gaussian(X, mu, sigma2):
    '''
    mu, sigma2参数已经决定了一个高斯分布模型


    一般高斯模糊
                              (x-u)^2
                   1       - ————————————
    p(x;u,𝝈^2) = —————— * ℮    2𝝈^2
                  √2π𝝈^2

    多元高斯模糊. 当协方差为0时的,即一般高斯模糊
                   1
    p(x;u,∑)  = —————————————— * exp(- 1/2 * (x-u)^T * ∑^-1 * (x - u))
                    n/2    1/2
                (2π) *  |∑|

    因为原始模型就是多元高斯模型在sigma2上是对角矩阵而已，所以如下：
    If Sigma2 is a matrix, it is treated as the covariance matrix.
    If Sigma2 is a vector, it is treated as the sigma^2 values of the variances
    in each dimension (a diagonal covariance matrix)
    output:
        一个(m, )维向量，包含每个样本的概率值。
    '''

    # 下面是不利用矩阵的解法，相当于把每行数据输入进去，不会出现内存错误。
    m, n = X.shape
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    # 函数的前半段 np.linalg.de其实这个函数就是为了计算方阵的行列式值的
    norm = 1. / (np.power((2 * np.pi), n / 2) * np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m, 1))
    for row in range(m):
        xrow = X[row]
        # 函数的后半段 np.linalg.inv 得到x的逆矩阵
        exp[row] = np.exp(-1 / 2 * ((xrow - mu).T).dot(np.linalg.inv(sigma2)).dot(xrow - mu))
    return norm * exp


def getGaussianParams(X, useMultivariate):
    """
    获取u和𝝈^2函数的值

    u = 1/m * ∑(x_i)

            1
    𝝈^2 = —————— ∑(x_i - u_i)^2
            m
    """
    mu = X.mean(axis=0)
    if useMultivariate:
        sigma2 = ((X - mu).T @ (X - mu)) / len(X)
    else:
        # numpy.var (arr, axis = None)： 计算指定数据 (数组元素)沿指定轴 (如果有)的方差。
        sigma2 = X.var(axis=0, ddof=0)
    return mu, sigma2


def plotContours(mu, sigma2):
    """
    画出高斯概率分布的图，在三维中是一个上凸的曲面。投影到平面上则是一圈圈的等高线。
    """
    delta = .3  # 注意delta不能太小！！！否则会生成太多的数据，导致矩阵相乘会出现内存错误。
    x = np.arange(0, 30, delta)
    y = np.arange(0, 30, delta)

    # 这部分要转化为X形式的坐标矩阵，也就是一列是横坐标，一列是纵坐标，
    # 然后才能传入gaussian中求解得到每个点的概率值
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]  # 按列合并，一列横坐标，一列纵坐标
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)  # 这步骤不能忘

    cont_levels = [10 ** h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, z, cont_levels)  # 这个levels是作业里面给的参考,或者通过求解的概率推出来。

    plt.title('Gaussian Contours', fontsize=16)


def selectThreshold(yval, pval):
    """
    确定哪些例子是异常的一种方法是通过一组交叉验证集，选择一个好的阈值 ε
    :param yval:
    :param pval:
    :return:
    """

    def computeF1(yval, pval):
        m = len(yval)
        """
        tp means true positives：是异常值，并且我们的模型预测成异常值了，即真的异常值。
        fp means false positives：是正常值，但模型把它预测成异常值，即假的异常值。
        fn means false negatives：是异常值，但是模型把它预测成正常值，即假的正常值。
        precision 表示你预测为positive的样本中有多少是真的positive的样本。
        recall 表示实际有多少positive的样本，而你成功预测出多少positive的样本。
        """
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return F1

    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpsilon = 0, 0
    for e in epsilons:
        pval_ = pval < e
        thisF1 = computeF1(yval, pval_)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpsilon = e

    return bestF1, bestEpsilon


if __name__ == "__main__":
    X, Xval, yval = read_data()
    # plot_data(X)
    # plt.show()

    ####################################################
    # 不使用多元高斯分布
    # plot_data(X)
    # useMV = False
    # plotContours(*getGaussianParams(X, useMV))
    # plt.show()
    #
    # ####################################################
    # # 使用多元高期分布
    # plot_data(X)
    # useMV = True
    # # *表示解元组
    # plotContours(*getGaussianParams(X, useMV))
    # plt.show()
    # ####################################################
    # 不使用多元高斯分布时，运算出来的准备率
    mu, sigma2 = getGaussianParams(X, useMultivariate=False)
    pval = gaussian(Xval, mu, sigma2)
    #yval实际异常值,pval异常的概率。越低越有可能是异常
    bestF1, bestEpsilon = selectThreshold(yval, pval)
    print(bestF1, bestEpsilon)

    y = gaussian(X, mu, sigma2)  # X的概率
    xx = np.array([X[i] for i in range(len(y)) if y[i] < bestEpsilon])
    print("异常点= ", xx)
    plot_data(X)
    plotContours(mu, sigma2)
    plt.scatter(xx[:, 0], xx[:, 1], s=80, facecolors='none', edgecolors='r')

    plt.show()
