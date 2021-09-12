# coding: utf-8 
# @时间   : 2021/8/25 8:16 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 6cv图像形态_腐蚀彭胀操作.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
腐蚀是形态学基本操作中的一种，它通过使图像沿着边界向内收缩，达到消除边界点的目的。
一般输入图像为二值图像。
腐蚀操作是将像素点与卷积核函数做与运算，只有全1时才保留这个像素点，否则为0，则舍弃。
"""
image = cv2.imread("./img/qianming.png")
k = np.ones((3, 3), np.uint8)
img = cv2.erode(image, k, iterations=2)
# cv2.imshow("原始", image)
cv2.imshow("腐蚀", img)
cv2.waitKey()
########################
img2 = cv2.dilate(img,(3,3),iterations=3)
cv2.imshow("膨胀", img2)
cv2.waitKey()
cv2.destroyAllWindows()


