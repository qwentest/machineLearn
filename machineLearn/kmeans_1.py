# coding: utf-8 
# @时间   : 2021/8/10 8:13 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 聚类算法的处理过程。
# @文件   : kmeans_1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


def read_mat():
    """
    读取源数据
    :return:
    """
    mat = loadmat('./data/ex7data2.mat')
    X = mat['X']
    return X


def findClosestCentroids(X, centroids):
    """
    根据簇中心，寻找最近的距离
    centroids，设定的簇中心
    """
    idx = []
    max_dist = 1000000  # 限制一下最大距离
    for i in range(len(X)):
        # 实际的是距离相减
        minus = X[i] - centroids
        # 因为这里出来的是一个2维数据，分别对两个值进行平方的求职
        dist = minus[:, 0] ** 2 + minus[:, 1] ** 2
        if dist.min() < max_dist:
            # 返回某行或某列的最小索引的下标
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)


def computeCentroids(X, idx):
    """
    分配好每个点对应的簇中心，接下来要做的是，重新计算每个簇中心，为这个簇里面所有点位置的平均值
    :param X:
    :param idx:
    :return:
    """
    centroids = []
    for i in range(len(np.unique(idx))):  # np.unique() means K
        u_k = X[idx == i].mean(axis=0)  # 求每列的平均值
        centroids.append(u_k)
    return np.array(centroids)


def runKmeans(X, centroids, max_iters):
    """
    运行k-means的主要逻辑。
                                 1
    𝐽(𝑐(1),...,𝑐(𝑚),𝜇1,...,𝜇k )= ————∑∥𝑋(𝑖)−𝜇(𝑖)∥^2
                                 m

    用𝜇1 ,𝜇 2 ,...,𝜇 𝑘 来表示聚类中心，用𝑐 (1) ,𝑐 (2) ,...,𝑐 (𝑚) 来存储与第𝑖个实例数据最近的聚类中
    心的索引，K-均值算法的伪代码如下:

    Repeat {
    for i = 1 to m
    c(i) := index (form 1 to K) of cluster centroid closest to x(i) for k = 1 to K
    μk := average (mean) of points assigned to cluster k
    }
    算法分为两个步骤
        第一个 for 循环是赋值步骤，即:对于每一个样例𝑖，计算其应该属于的类。
        第二个 for 循环是聚类中心的移动，即:对于每一个类𝐾，重新计算该类的质心
    :param X:
    :param centroids: 簇中心的初始值
    :param max_iters: 运行的次数
    :return:
    """
    # K = len(centroids)
    centroids_all = []
    centroids_all.append(centroids)
    centroid_i = centroids
    for i in range(max_iters):
        # 获取距离
        idx = findClosestCentroids(X, centroid_i)
        # 获取平均值
        centroid_i = computeCentroids(X, idx)
        centroids_all.append(centroid_i)
    return idx, centroids_all


def plotData(X, centroids, idx=None):
    """
    可视化数据，并自动分开着色。
    idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    centroids: 包含每次中心点历史记录
    """
    colors = ['b', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab', 'maroon', 'navy', 'sienna', 'tomato', 'lightgray',
              'gainsboro', 'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']

    assert len(centroids[0]) <= len(colors), 'colors not enough '

    subX = []  # 分号类的样本点
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)
    else:
        subX = [X]  # 将X转化为一个元素的列表，每个元素为每个簇的样本集，方便下方绘图
    # 分别画出每个簇的点，并着不同的颜色
    plt.figure(figsize=(8, 5))
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label='Cluster %d' % i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.title('Plot of X Points', fontsize=16)

    # 画出簇中心点的移动轨迹
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:, 0])
        yy.append(centroid[:, 1])

    plt.plot(xx, yy, 'rx--', markersize=8)


def initCentroids(X, K):
    """随机初始化"""
    m, n = X.shape
    idx = np.random.choice(m, K)
    centroids = X[idx]

    return centroids


if __name__ == "__main__":
    X = read_mat()
    init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    # idx = findClosestCentroids(X, init_centroids)
    # print(idx)
    # print(idx[0:3])
    # computeCentroids(X, idx)

    # 初始簇中心的位置
    # plotData(X, [init_centroids])
    # plt.show()

    idx, centroids_all = runKmeans(X, init_centroids, 20)
    plotData(X, centroids_all, idx)
    plt.show()

    # 不同的随机样本，位置不同。

    # for i in range(3):
    #     centroids = initCentroids(X, 3)
    #     idx, centroids_all = runKmeans(X, centroids, 10)
    #     plotData(X, centroids_all, idx)
    #     plt.show()
