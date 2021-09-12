# coding: utf-8 
# @时间   : 2021/9/10 9:29 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 18_himmelblau函数优化.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
import matplotlib.pyplot as plt


def himmelblau(x):
    """
        https://blog.csdn.net/VincentWeng1/article/details/114129081
        f(x,y) = (x^2 + y -11)^2 + (x + y^2 -7)^2
        himmelblau函数是数学家们构造出来的一个特殊的函数，可以用来测试深度学习算法是否能够收敛到局部最小值
        这个函数最小值为0，有四个最小值点，分别是：

        f(3.0,2.0)=0
        f(−2.805118,3.131312)=0
        f(−3.779310,−3.283186)=0
        f(3.584428,−1.848126)=0
        """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == '__main__':
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)  #
    z = himmelblau([X, Y])
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z)
    ax.view_init(60, -30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
