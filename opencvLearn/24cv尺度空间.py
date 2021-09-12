# coding: utf-8 
# @时间   : 2021/9/6 9:31 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 24cv尺度空间.py

# SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。
# SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等。
# https://blog.csdn.net/yan_520csdn/article/details/101349493
# https://blog.csdn.net/yukinoai/article/details/88912586

# 1。特征匹配的方法：
#    1.暴力匹配，即直接对比，采用欧式距离来计算。
#    2.一对一匹配
#    3.Knn值匹配
#    4.RANSAC算法匹配



import numpy as np
import cv2

img = cv2.imread('./img/dog.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
创建一个SIFT对象:
cv2.xfeatures2d.SIFT_create(, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
nfeatures：默认为0，要保留的最佳特征的数量。 特征按其分数排名（在SIFT算法中按局部对比度排序）
nOctaveLayers：默认为3，金字塔每组(Octave)有多少层。 3是D. Lowe纸中使用的值。
contrastThreshold：默认为0.04，对比度阈值，用于滤除半均匀（低对比度）区域中的弱特征。 阈值越大，检测器产生的特征越少。
edgeThreshold：默认为10，用来过滤边缘特征的阈值。注意，它的意思与contrastThreshold不同，edgeThreshold越大，滤出的特征越少（保留更多特征）。
sigma：默认为1.6，高斯金字塔中的σ。 如果使用带有软镜头的弱相机拍摄图像，则可能需要减少数量。

"""
sift = cv2.xfeatures2d.SIFT_create()
"""
检测特征点:

sift.detect(image,keypoints)  或  keypoint = sift.detect(image, None)

sift：配置好SIFT算法对象
image：输入图像，单通道
keypoint：输出参数，保存着特征点，每个特征点包含有以下信息：
"""
kp = sift.detect(gray, None)

#绘制关键点
img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('figure', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

