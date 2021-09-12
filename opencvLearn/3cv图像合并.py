# coding: utf-8 
# @时间   : 2021/8/21 5:02 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 图像融合
# @文件   : 3cv图像合并.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./img/cat.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('./img/dog.png', cv2.IMREAD_COLOR)
img1_new = cv2.resize(img1, (470, 400))  # 改变尺寸，使得维度相同。也可以按倍数来进行改变尺寸
print(img1_new.shape, img2.shape)

img = cv2.addWeighted(img1_new, 0.4, img2, 0.6, 0)  # 两个图像融合 ，即 θ_1 * x1 + θ_2 * x2 + b，0即偏值项
cv2.imshow('title', img)  # show
cv2.waitKey(0)  # 多少秒自动消灭，0表示任意key退出
cv2.destroyAllWindows()  # 释放句柄
