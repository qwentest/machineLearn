# coding: utf-8 
# @时间   : 2021/8/30 2:40 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 19cv图像直方图均衡化.py

"""
直方图均衡化, 把直方图的每个灰度级进行归一化处理，求每种灰度的累积分布，得到一个映射的灰度映射表，
然后根据相应的灰度值来修正原图中的每个像素.
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_equalizehist_img(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)

    plt.subplot(221)
    plt.imshow(img, plt.cm.gray)
    plt.axis('off')

    plt.subplot(222)
    plt.hist(img.ravel(), 256)

    plt.subplot(223)
    plt.imshow(equ, plt.cm.gray)
    plt.axis('off')

    plt.subplot(224)
    plt.hist(equ.ravel(), 256)

    plt.savefig("result2.jpg")


get_equalizehist_img('./img/girl.png')
