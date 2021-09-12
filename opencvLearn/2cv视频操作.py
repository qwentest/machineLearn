# coding: utf-8 
# @时间   : 2021/8/21 4:18 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 视频文件中的图像读取和处理
# @文件   : 2cv视频操作.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

vc = cv2.VideoCapture('./img/test.mov')  # 读取视频文件或者摄像头
if vc.isOpened():
    open1, frame = vc.read()  # 读取1帧的图像.open是否有图像,frame这一帧的图像值
else:
    open1 = False

while open1:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转换成一个灰度图
        cv2.imshow('result', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
