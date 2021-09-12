# coding: utf-8 
# @时间   : 2021/8/29 2:46 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 13cv_Canny边缘检测.py
"""
https://www.jianshu.com/p/effb2371ea12
Canny算子是由计算机科学家John F. Canny于1986年提出的一种边缘检测算子，是目前理论上相对最完善的一种边缘检测算法。
Canny边缘检测的算法步骤：
1.使用高斯滤波器，以平滑图像，滤除噪声。
2.计算图像中每个像素点的梯度强度和方向(选择一个算子）。
3.应用非极大值抑制，以消除边缘检测带来的杂散响应。
4.应用双阈值检测来确定真实和潜在的边缘。
5.通过抑制孤立弱边缘最终完成边缘检测。
"""
# coding=utf-8
import cv2
import numpy as np

# 边缘检测都是针对灰度图像
img = cv2.imread("./img/girl.png", cv2.IMREAD_GRAYSCALE)
# 高斯模糊，可以尽量保留轮廓
img = cv2.GaussianBlur(img, (3, 3), 1)
# 在50至150之间的保留，否则丢失
canny = cv2.Canny(img, 50, 150)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
