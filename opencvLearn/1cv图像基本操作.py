# coding: utf-8 
# @时间   : 2021/8/21 3:13 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : opencv对图像的基本操作
# @文件   : 1cv图像基本操作.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
在cv2中图像显示顺序是BGR格式，所以是3个维度
图像值的范围是0-255，如果是灰度图则只有1维
cv2.IMREAD_COLO, BGR
cv2.IMREAD_GRAYSCALE，灰度图
"""
img = cv2.imread('./img/girl.png', cv2.IMREAD_COLOR)  # read
print(img.shape, img)
cv2.imshow('title', img)  # show
cv2.waitKey(0)  # 多少秒自动消灭，0表示任意key退出
cv2.destroyAllWindows()  # 释放句柄
cv2.imwrite('./img/cat.png', img)  # 写入图像
##############################################
# # 截取部分图像区域中的内容
cat = img[0:200, 0:200]
cv2.imshow('变小了', cat)
cv2.waitKey(0)
cv2.destroyAllWindows()
###############################################
# 按通道进行分割
b, g, r = cv2.split(img)
print(b.shape)
# 按通道合并
img = cv2.merge((b, g, r))
img_copy = img.copy()
# # 只保留b
img_copy[:, :, 1] = 0
img_copy[:, :, 2] = 0
cv2.imshow('b', img_copy)
cv2.waitKey(0)
# 只保留g
img_copy = img.copy()
img_copy[:, :, 0] = 0
img_copy[:, :, 2] = 0
cv2.imshow('g', img_copy)
cv2.waitKey(0)
# 只保存r
img_copy = img.copy()
img_copy[:, :, 0] = 0
img_copy[:, :, 1] = 0
cv2.imshow('r', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
###############################################
# 边界填充 ，即四个方向变大处理
top, bottom, left, right = (50, 50, 50, 50)
"""
borderType：边界的类型
BORDER_REPLICATE：复制法，即复制最边缘的像素。例如：aaaa|abcdefg|ggggg
BORDER_REFLECT：反射法,即以最边缘的像素为对称轴。例如：fedcba|abcdefg|gfedec
BORDER_REFLECT_101：反射法,也是最边缘的像素为对称轴，但与BORDER_REFLECT有区别。例如：fedcb|abcdefg|fedec
BORDER_WRAP：外包装法，即以图像的左边界与右边界相连，上下边界相连。例如：cdefgh|abcdefgh|abcdefg
BORDER_CONSTANT：常量法。
"""
img_make = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=cv2.BORDER_REPLICATE)
cv2.imshow('copy',img_make)
cv2.waitKey(0)
cv2.destroyAllWindows()