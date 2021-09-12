# coding: utf-8 
# @时间   : 2021/9/3 11:25 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 20cv傅立叶变换.py
"""
高频：变化剧烈的灰度分量；
低频：变化缓慢的灰度分量；
低通滤波器：只保留低频，会使得图像模糊；
高通滤波器：只保留高频，会使得图像细节增强。
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
cv2.dft()转成频率图，cv2.idft()转换成图像。
得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
cv2.dft()得到的是双通首的（实部、虚部），通常还需要转换成图像格式才能展示（0,255)。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./img/lenaNoise.png", 0)
# opencv要求为np.float32格式
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# 实现频率的转换，将低频转换到中心点
dftShift = np.fft.fftshift(dft)
# 实现频率到图像的转换
result = 20 * np.log(cv2.magnitude(dftShift[..., 0], dftShift[..., 1]))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('result')
plt.axis('off')

plt.show()
