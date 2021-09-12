# coding: utf-8 
# @时间   : 2021/8/27 2:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 8cv图像形态_梯度运算_边缘轮廓.py
#梯度 = 膨胀 - 腐蚀，得到边缘轮廓
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 所谓开运算，就是先腐蚀后膨胀；
image = cv2.imread("./img/qianming.png")
kernel = np.ones((3, 3))
img = cv2.morphologyEx(image,cv2.MORPH_GRADIENT,kernel)
cv2.imshow("梯度", img)
cv2.waitKey()
cv2.destroyAllWindows()