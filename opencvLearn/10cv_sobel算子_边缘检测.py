# coding: utf-8 
# @时间   : 2021/8/27 2:53 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 10cv_sobel算子_边缘检测.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
https://blog.csdn.net/weixin_41500849/article/details/80611263
Sobel算子根据像素点的上下、左右相邻点的灰度加权差，在边缘达到极值的现象来检测边缘。Sobel算子是一种常用的边缘检测方法。
可以用它来表明其清晰度。

在Sobel函数的第二个参数这里使用了cv2.CV_16S。
因为OpenCV文档中对Sobel算子的介绍中有这么一句：
“in the case of 8-bit input images it will result in truncated derivatives”。
即Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。convertScaleAbs()的原型为：
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来。
其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
"""
img = cv2.imread("./img/cat.png")
kernel = np.ones((5, 5))
#                              水平方向
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
#                              垂直方向
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("orign", img)
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
