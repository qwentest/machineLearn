# coding: utf-8 
# @时间   : 2021/8/29 2:25 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 11cv_Scharr算子_边缘检测.py
"""
Sobel 算子虽然可以有效的提取图像边缘，但是对图像中较弱的边缘提取效果较差。
Scharr 算子的主要思路是通过将模版中的权重系数放大来增大像素值间的差异。
https://www.cnblogs.com/babycomeon/p/13282389.html
"""

import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./img/girl.png")
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Scharr 算子
x = cv.Scharr(gray_img, cv.CV_16S, 1, 0) # X 方向
y = cv.Scharr(gray_img, cv.CV_16S, 0, 1) # Y 方向
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
Scharr = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

# 显示图形
plt.rcParams['font.sans-serif']=['SimHei']

titles = ['old', 'Scharr']
images = [rgb_img, Scharr]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()