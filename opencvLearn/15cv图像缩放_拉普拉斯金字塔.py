# coding: utf-8 
# @时间   : 2021/8/29 3:28 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 15cv图像缩放_拉普拉斯金字塔.py
"""
拉普拉斯金字塔是高斯金字塔的修正版，为了还原到原图。通过计算残差图来达到还原
"""
import cv2
import numpy as np

img = cv2.imread("./img/katong.png")
down = cv2.pyrDown(img)
up = cv2.pyrUp(down)
result = img - up
cv2.imshow('Canny', up)
cv2.waitKey(0)
cv2.destroyAllWindows()
