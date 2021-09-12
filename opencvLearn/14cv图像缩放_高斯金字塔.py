# coding: utf-8 
# @时间   : 2021/8/29 3:10 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 14cv图像缩放_高斯金字塔.py
"""
https://blog.csdn.net/xiachong27/article/details/88853384

一幅图像的金字塔是一系列以金字塔形状排列的分辨率逐步降低，且来源于同一张原始图的图像集合。
其通过梯次向下采样获得，直到达到某个终止条件才停止采样。
金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似。
我们将一层一层的图像比喻成金字塔，层级越高，则图像越小，分辨率越低。
"""
import cv2
import numpy as np

img = cv2.imread("./img/katong.png")
up = cv2.pyrUp(img)  # 向上金字搭，变大
# up = cv2.pyrDown(img)  # 向下金字塔，变小
cv2.imshow('Canny', up)
cv2.waitKey(0)
cv2.destroyAllWindows()
