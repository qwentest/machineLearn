# coding: utf-8 
# @时间   : 2021/8/29 2:33 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 12cv_lapkacian算子_边缘检测.py
"""
1、Laplacian算子通过对邻域中心像素的四个方向或八个方向寻求梯度。
然后将梯度加起来判断中心像素灰度与邻域中其他像素灰度的关系，最后通过梯度运算的结果调整像素灰度。
2、分为四邻域和八邻域，四邻域是邻域中心像素的四个方向求梯度，八邻域是八个方向求梯度。
"""
import cv2
import numpy as np

img = cv2.imread("./img/cat.png", 0)

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap)

cv2.imshow('laplacian', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

