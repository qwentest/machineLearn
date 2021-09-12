# coding: utf-8 
# @时间   : 2021/9/3 8:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 23cv角点检测.py

"""
水平和垂直方向，变化很大的方框（区域），就是角点。特征更加明显。
https://juejin.cn/post/6992167968995606558

角点所具有的特征：
1。轮廓之间的交点
2。对于同一场景，即使视角发生变化，通常具备稳定性质的特征
3。该点附近区域的像素点无论在梯度方向上还是其梯度幅值上有着较大变化

原理：使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度
变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点.

cv2.cornerHarris（）
src 必须是float32
block_size 邻域大小，越大表示用更粗的点标记角点
ksize Sobel求导中的窗口大小
k 自由参数，取值[0.04, 0.06]

"""
import cv2
import numpy as np

chess = cv2.imread('./img/qipan.PNG')
gray = cv2.cvtColor(chess, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(gray, 2,  # 设定值
                       3, 0.04)  # 推荐值
chess[dst > 0.1 * np.max(dst)] = [0, 0, 255]
cv2.imshow('figure', chess)
cv2.waitKey(0)
cv2.destroyAllWindows()
