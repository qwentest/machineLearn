# coding: utf-8 
# @时间   : 2021/8/22 3:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 什么是图像分割：
#           1.图像分割是指将图像分成若干具有相似性质的区域的过程.
#           2.主要有基于阈值、基于区域、基于边缘、基于聚类、基于图论和基于深度学习的图像分割方法等。
#           3.图像分割分为语义分割和实例分割
# @文件   : 4cv图像分割_基于阈值.py
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./img/cat.png', 0)
# retval, dst	= cv2.threshold( src, thresh, maxval, type[, dst] )
# src：输入图像，单通道或四通道图像。
# thresh：设定的固定阈值。
# maxval：当type参数设置为THRESH_BINARY时，表示像素值大于阈值时设置的值，或设置为THRESH_BINARY_INV时，
# 表示像素值小于阈值时设置的值。
# type：阈值类型
# cv.THRESH_BINARY ,   超过部分取255，否则0；
# cv.THRESH_BINARY_INV，超过部分为0，否则255;
# cv.THRESH_TRUNC,      大于阈值部分设为255，否则不变
# cv.THRESH_TOZERO,     大于阈值部分为0，否则不变

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
# matplotlib 中文全局处理.
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
