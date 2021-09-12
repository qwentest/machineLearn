# coding: utf-8 
# @时间   : 2021/8/8 11:54 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 高斯核SVM
# @文件   : SVMgaussian_2.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

"""
用来处理非线性分类器的支持向量机，要使用到核函数的相关算法。
前面我们知道：为了非线型区分，我们可以增加多项式，如：
if h(θx) = 1，则
    θ_0  + θ_1 * x1  + θ_2 * x2 + θ_3 * x1 * x2 + θ_4 * x1^2 + θ_5 * x2^2 >= 0
多项式的增加，容易出现过拟合的情况，而且随着特征的增加，其计算量也会突增

令：f1 = x1,f2 = x2,f3 = x1 * x2, f4 =x1^2,f5 = x2 ^2,则
   θ_0  + θ_1 * f1  + θ_2 * f2 + θ_3 * f3 + θ_4 * f4 + θ_5 * f5 >= 0,
   我们定义3个新的特征,使用l1,l2,l3来进行定义[L是我们标记的位置],如下图所示.
   kernel
   
   X2   │
        │
        │           
        │  
        │       .l1
        │         .x
        │  .l2       
        │    .x        .l3
        │               .x
        │  
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>X1
        
                                    
    则                            |x-l1|^2
    f1 = similarty(x, l1) = exp(- ————————)
                                     2𝝈^2
                                  |x-l2|^2             
    f2 = similarty(x, l2) = exp(- ————————)
                                     2𝝈^2
                n
                ∑ |x-l2|^2 
                j=1            
    f  = exp(- ————————)
                2𝝈^2
                
    判断某个样本是正样本还是负样本,取决于x与L(i)这个标记的欧式距离,
    即相似度越高则欧式距离为0则f(1) 趋近于 1,如果距离越远则f(1) 趋近于0,
    这个函数我们又称之为核函数,这里称之为高斯核函数
                             0^2
    如果x ≈ l1，则f1 ≈ exp(- ——————) ≈ 1
                            2𝝈^2
                                large^2
    如果x 离l1 越远，则f1 ≈ exp(- ——————————) ≈ 0
                                 2𝝈^2
    如上图l点，假设θ_0 = -0.5,θ_1 = 1, θ_2 = 0, θ_3 = 0
    θ_0  + θ_1 * f1  + θ_2 * f2 + θ_3 * f3  >= 0
    则, -0.5 + f1  > = 0，如果要成立，则我们可以设f1 = 1,又f1 = x1，则我们只需寻找l1的值即可。
    -------------------------------------------------------------------------------
    如何寻找l_i？
    假设(x_1,y_1)(x_2,y_2),....(x_m,y_m)
    设l_1 = x_1,l_2 = x_2,....l_m = x_m
    则,
    f_1 = similarty(x, l_1) = similarty(x, x_1)
    f_2 = similarty(x, l_2) = similarty(x, x_2)
    所以假设有(x_i,y_i)的样本,则f_i的计算过程就是每一个x_i距离其它x_i的距离，并且其中有一个距离是x_i = x_i时
                  0
    f_i = exp(- ———————) = 1
                 2𝝈^2
                 
    则f_i向量化后为R^n+1维的向量。
    [ f_0^i
      f_1^i
      ...
      f_n^i 
    ]
    
    所以代入支持向量机中的公式就变成了
    predict y = 1 ，则θ^T f >= 0
            m                                                    1  n
    C  *   [∑ y * cost_0(θ^T * f^i ) + (1 - y) cost_1(θ^T * f^i)] + --- ∑ θ_j^2 
            i=1                                                  2  j=1
    其中 C = 1 / 𝜆, C越小,就可能低偏差,高方差,可能出现过拟合问题;
                   C越大,就可能高偏差,低方差,可能出现欠拟合的问题.
                   
    如果是多个特征,但是样本数据较少,则选择线性SVM;
    如果特征较少,样本较多,则选择高斯核SVM;
    
    
"""


def gaussKernel(x1, l2, sigma):
    """
    核函数的数据表示方法
    :param x1:
    :param x2:
    :param sigma:
    :return:
    """
    return np.exp(- ((x1 - l2) ** 2).sum() / (2 * sigma ** 2))


def readSVMData(isShow=True):
    """
    将数据源中的数据，图形化展示
    :return:
    """
    mat = loadmat('./data/ex6data2.mat')
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


def svm_gaussian(X, y):
    """
    高斯核SVM的区分
    :param X:
    :param y:
    :return:
    """
    sigma = 0.1
    gamma = np.power(sigma, -2) / 2
    clf = svm.SVC(C=1, kernel='rbf', gamma=gamma)
    modle = clf.fit(X, y.flatten())
    return modle


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


def svm_gaussina_split(model):
    """"
    图示化，不同的C值，间隙距离的展示。
    """
    mat = loadmat('./data/ex6data2.mat')
    X = mat['X']
    y = mat['y']
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    print("间隔 = ", model.decision_function(X))
    plotBoundary(model, X)
    plt.title('svm split')
    plt.show()

def getBestC_SVMSplit():
    """
    如果有多个C值，取最优评分中的C值。
    :return:
    """
    mat3 = loadmat('./data/ex6data3.mat')
    X, y = mat3['X'], mat3['y']
    Xval, yval = mat3['Xval'], mat3['yval']
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

    """尝试多个C值，并给C值通过SVM出来的模型给预评分。"""
    Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
    sigmavalues = Cvalues
    best_pair, best_score = (0, 0), 0

    for C in Cvalues:
        for sigma in sigmavalues:
            gamma = np.power(sigma, -2.) / 2
            model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            model.fit(X, y.flatten())
            this_score = model.score(Xval, yval)
            if this_score > best_score:
                best_score = this_score
                best_pair = (C, sigma)
    print('best_pair={}, best_score={}'.format(best_pair, best_score))

    model = svm.SVC(C=best_pair[0], kernel='rbf', gamma=np.power(best_pair[1], -2.) / 2)
    model.fit(X, y.flatten())

    mat3 = loadmat('./data/ex6data3.mat')
    X, y = mat3['X'], mat3['y']
    #Xval, yval = mat3['Xval'], mat3['yval']
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    #决策边界
    plotBoundary(model, X)
    plt.show()


if __name__ == "__main__":
    pass
    # data = readSVMData(isShow=False)
    # clfs = svm_gaussian(data[0],data[1])
    # svm_gaussina_split(clfs)
    # getBestC_SVMSplit()
