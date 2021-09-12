# coding: utf-8 
# @时间   : 2021/8/27 2:25 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 9cv图像形态_礼帽黑帽.py
# 所谓，礼帽 = 原始输入 - 开运算的结果（开腐蚀后膨胀）
#      黑帽 = 闭运算（先膨胀后腐蚀) - 原始输入
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./img/qianming.png")
kernel = np.ones((5, 5))
#礼帽留下毛剌 cv.MORPH_TOPHAT
img = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,kernel)
#黑帽留下轮阔
cv2.imshow("礼帽", img)
cv2.waitKey()
cv2.destroyAllWindows()