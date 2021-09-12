# coding: utf-8 
# @时间   : 2021/9/3 2:23 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 21cv傅立叶变换_高通滤波.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./img/lenaNoise.png", 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 中间值设置为0，即低频率的设置为0
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
# 选择了30 * 30的区域
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

# 实现滤波的逆转换成图像
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(iimg, cmap='gray')
plt.title('result')
plt.axis('off')

plt.show()
