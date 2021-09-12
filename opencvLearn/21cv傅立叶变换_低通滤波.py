# coding: utf-8 
# @时间   : 2021/9/3 2:30 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 21cv傅立叶变换_低通滤波.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./img/lenaNoise.png", 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

rows,cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
mask = np.zeros((rows, cols, 2), np.uint8)
#将高频率的设置为1
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift = dftShift * mask
ishift = np.fft.ifftshift(fshift)
iimg = cv2.idft(ishift)
iimg = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(iimg, cmap = 'gray')
plt.title('result')
plt.axis('off')

plt.show()

