# coding: utf-8 
# @时间   : 2021/8/5 10:52 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 线型SVM
# @文件   : SVMline_1.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm


def sigmoid(z):
    """
    逻辑回归的激活函数，其中z = θ^T * X = θ_0 * x_0 + θ_1 * x_1+θ_2 * x_2 + ..θ_n * x_n
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(- z))


"""
    如果我们只考虑一个样本，则逻辑回归的损失函数为
    -y * log hθ(x) - (1 - y )* log 1- hθ(x)
    代入激活函数，则为
                1                                     1
    -y * log —————————————  - ( 1- y ) * log 1 - ——————————————
              1 + e ** -z                          1 + e ** -z
    将-号代入一下，就变成
                 1                                    1
    y * -log —————————————  + ( 1- y ) * -log 1 - ——————————————
              1 + e ** -z                          1 + e ** -z
    假设y = 1，则
                    1        
    cost_0= -log —————————————  
                  1 + e ** -z  
    因为需要y = 1，则z >> 0。z = 1为区分，视 z >= 1为一条折线，而且只有z >= 1时，Costθ才可能为0
    假设y = 0，则
                            1        
    cost_1= -log 1 -  —————————————  
                       1 + e ** -z 
    因为需要y = 0，则z << 0。z = -1为区分，视 z <= -1为一条折线，而且只有z <= -1时，Costθ才可能为0
    
    所以代入原损失函数得到
           m                                                     𝜆  n
    1/m *[ ∑  y * cost_0(θ^T * X ) + (1 - y) cost_1(θ^T * X)] + --- ∑ θ_j^2 
           i=1                                                   2m j=1
    去掉1/m，则
    m                                                     𝜆  n
    [ ∑y * cost_0(θ^T * X ) + (1 - y) cost_1(θ^T * X)] + --- ∑ θ_j^2 
    i=1                                                   2  j=1
    
    令                        (1 - y) cost_1(θ^T * X)
        A = cost_0(θ^T * X) + ----------------------
                                    2
        B = θ_j^2
                    1
    则  A + 𝜆 * B = --- * A + B
                    𝜆
    原因：假如(u-5)^2 + 1，      则u只能等于5
            10 * (u-5)^2 + 10，则u仍然只能等于5
    所以令C = 1/𝜆，所以Jθ = C * A + B，所以
            m                                                    1  n
    C  *   [∑ y * cost_0(θ^T * X ) + (1 - y) cost_1(θ^T * X)] + --- ∑ θ_j^2 
            i=1                                                  2  j=1
    需要注意,如果是y=1的正样本,我们需要令𝝷^Tx>>1;
    如果是y=0的负样本,则我们需要令𝝷^Tx<< -1,那么在[1,-1]之间我们其实是有一个间隙的空间.
    Costθ
        │
        │
        │                       O
        │  ↘  ↘︎  ↘        O   O
        │    ↘  ↘︎  ↘      O
        │  X X ↘   ↘︎  ↘      O   O
        │    X   ↘   ↘︎  ↘  O  O   O   
        │   X   X   ↘   ↘︎  ↘       O   O
     0.5│  X  X  X     ↘   ↘  ↘
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>z
        -2    -1   0  1   2   3
    
    通过SVM算法,会产生黑色这个区分线(正负样本),拥有同时离正负样本最小的距离,这个距离又称之为支持向量机的间距.
    这使用SVN会拥有鲁棒性,会尽量用大的间距去分离不同的样本.

"""


def log_sigmoid_1(z):
    """
    当y = 1，则Costθ与z之间的关系，大概如cost_0()函数所示。
    因为要y = 1, 所以θ^T * X >> 0 ,则将z = 1为区分，视 z >= 1为一条折线，而且只有z >= 1时，Costθ才可能为0

    Costθ
        │
        │
        │
        │   ↘︎
        │     ↘︎
        │       ↘︎
        │         ↘︎
        │           ↘︎
     0.5│             ↘︎_________
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>z
        -2    -1   0  1   2   3
    :param z:
    :return:
    """
    return -np.log(sigmoid(z))


def log_sigmoid_0(z):
    """
     当y = 0，，则Costθ与z之间的关系，大概如cost_1()函数所示。
     因为要y = 0, 所以θ^T * X << 0 ,则将z = -11为区分，视 z <= -1为一条折线，而且只有z <= -1时，Costθ才可能为0
     Costθ
        │
        │
        │                        ↙
        │                      ↙
        │                    ↙
        │                  ↙
        │        ︎        ↙
        │             ↙               ︎
     0.5│  ︎_________↙︎
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>z
        -3  -2    -1   0  1   2   3

    :param z:
    :return:
    """
    return -np.log(1 - sigmoid(z))


def show(z, cost):
    fig, ax = plt.subplots()
    plt.xlabel('z')
    plt.ylabel('cost')
    # ax.set_yticks(cost)
    ax.set_xticks(z)
    plt.plot(z, cost)
    plt.plot([0.5] * len(z))
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


def cost_0():
    ll = [-3, -2, -1, 0, 1, 2, 3]
    z = np.array(ll)
    y1 = log_sigmoid_0(z)
    show(z, y1)


def cost_1():
    ll = [-3, -2, -1, 0, 1, 2, 3]
    z = np.array(ll)
    y1 = log_sigmoid_1(z)
    show(z, y1)


def readSVMData(isShow=True):
    """
    将数据源中的数据，图形化展示
    :return:
    """
    mat = loadmat('./data/ex6data1.mat')
    X = mat['X']
    y = mat['y']
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    if isShow:
        plt.show()
    return X, y


def plotBoundary(clf, X):
    """
    SVM的线性决策边界
    :param clf:
    :param X:
    :return:
    """
    x_min, x_max = X[:, 0].min() * 1.2, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 1.1, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)


def svm_line(X, y):
    """
    线型SVM，尝试了不同的C值1和100
    :param X:
    :param y:
    :return:
    """
    # svc参数的说明：https://www.jianshu.com/p/a9f9954355b3
    models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
    # 根据给定的训练数据拟合SVM模型。
    clfs = [model.fit(X, y.ravel()) for model in models]
    #clf.predict(new)，即使用模型来预测数据的分类结果。
    return clfs


def svm_line_show(clfs):
    """"
    图示化，不同的C值，间隙距离的展示。
    """
    title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
    for model, title in zip(clfs, title):
        mat = loadmat('./data/ex6data1.mat')
        X = mat['X']
        y = mat['y']
        plt.figure(figsize=(8, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        print("间隔 = ", model.decision_function(X))
        plotBoundary(model, X)
        plt.title(title)
        plt.show()



if __name__ == "__main__":
    data = readSVMData(isShow=False)
    clfs = svm_line(data[0], data[1])
    svm_line_show(clfs)
