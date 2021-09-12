# coding: utf-8 
# @时间   : 2021/8/27 2:13 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 7cv图像形态_开闭运算.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 所谓开运算，就是先腐蚀后膨胀；
image = cv2.imread("./img/qianming.png")
kernel = np.ones((5, 5))
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow("开运算", opening)
cv2.waitKey()
cv2.destroyAllWindows()
# 所谓闭运算，就是先膨胀后腐蚀；
# opening = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
