# coding: utf-8 
# @时间   : 2021/8/30 9:04 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 16cv图像轮廓.py
"""
https://blog.csdn.net/wumu720123/article/details/89646777
边缘检测能够测出边缘，但是边缘是不连续的，将边缘连接为一个整体，构成轮廓
注意事项：
1.对象是二值图像。所以需要预先进行阈值分割或者边缘检测处理。
2.查找轮廓需要更改原始图像。因此，通常使用原始图像的一份拷贝操作。
3.在OpenCV中，是从黑色背景中查找白色图像。因此，对象必须是白色的，背景必须是黑色的。
"""
import cv2
import numpy as np
o = cv2.imread("./img/yuan.png")
cv2.imshow("original", o)
co = o.copy()
gray = cv2.cvtColor(co, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 一般为cv2.RETR_TREE
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
r = cv2.drawContours(co, contours, -1, (0, 0, 255), 1)
cv2.imshow("result", r)
cv2.waitKey(0)
cnt = contours[1]
# 轮廓特征
# 面积
print(cv2.contourArea(cnt))
# 周长,第二个参数指定图形是否闭环,如果是则为True, 否则只是一条曲线.
print(cv2.arcLength(cnt, True))

# 轮廓近似，epsilon数值越小，越近似
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
tmp2 = o.copy()
# 注意，这里approx要加中括号
res3 = cv2.drawContours(tmp2, [approx], -1, (0, 0, 255), 2)
cv2.imshow("res3", res3)
cv2.waitKey(0)
# 外接图形 框起来
x, y, w, h = cv2.boundingRect(cnt)
# 直接在图片上进行绘制，所以一般要将原图复制一份，再进行绘制
tmp3 = o.copy()
res4 = cv2.rectangle(tmp3, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('rectangle', res4)
cv2.waitKey(0)
cv2.destroyAllWindows()