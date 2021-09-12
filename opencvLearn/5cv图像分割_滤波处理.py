# coding: utf-8 
# @时间   : 2021/8/22 3:39 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 图像滤波则通过运算来排除图像中和周围相差大的像素。
#          主要作用：过滤可以移除图像中的噪音、提取感兴趣的可视特征、允许图像重采样，使得图像模糊、去噪、平滑。
# @文件   : 5cv图像分割_滤波处理.py
# 各种滤波的总结：https://blog.csdn.net/qq_27261889/article/details/80822270
import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('./img/girl.png')


# 均值滤波
# 指在图像上对目标像素给一个模板，再用模板中的全体像素的平均值来代替原来像素值
# 假设图像值为：
# 23   158 140 115 131 87     131
# 238  0   67  16  247 14     220
# 199  |197 25  106 156 159|  173
# 94   |149 40  107 5   71 |  171
# 210  |163 198 226 223 156|  159
# 107  |222 37  68  193 157|  110
# 255  |42  72  250 41  75 |  184
# 77   150 17  248 197 147    150
# 218  235 106 128 65  197    202
# 其中红色(226)区域的像素值均值滤波处理过程为：
# ((197+25+106+156+159)+ (149+40+107+5+71)+ (163+198+226+223+156)+ (222+37+68+193+157)+ (42+72+250+41+75)) / 25 = 125.52
# 优点：
# 　　效率高，思路简单
# 缺点：
# 　　均值滤波本身存在着固有的缺陷，即它不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节，从而使图像变得模糊，不能很好地去除噪声点，特别是椒盐噪声。
def blur_filter_func(filename):
    img = cv2.imread(filename)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 均值滤波: 简单的平均卷积操作
    result = cv2.blur(img, (5, 5))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # 显示图像
    titles = ['origin image', 'blur image']
    images = [rgbimg, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 方框滤波: 基本和均值滤波一样，可以选择归一化，如果选择归一化，则和均值滤波一样
# 如果不选择归一化，则方框滤波容易越界，越界的话，超过255则用255表示
def box_blur_filter_func(filename):
    img = cv2.imread(filename)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blur = cv2.blur(img, (3, 3))
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    # 方框滤波: 基本和均值滤波一样，可以选择归一化，如果选择归一化，则和均值滤波一样
    # 如果不选择归一化，则方框滤波容易越界，越界的话，超过255则用255表示
    result_normal = cv2.boxFilter(img, -1, (5, 5), normalize=True)
    result_normal = cv2.cvtColor(result_normal, cv2.COLOR_BGR2RGB)

    result_nonormal = cv2.boxFilter(img, -1, (5, 5), normalize=False)  # 越界则用255来表示
    result_nonormal = cv2.cvtColor(result_nonormal, cv2.COLOR_BGR2RGB)
    # 显示图像
    titles = ['origin image', 'boxFilter image no normalize', 'boxFilter image normalize', 'blut filter image']
    images = [rgbimg, result_nonormal, result_normal, blur]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def gaussian_blur(path):
    """高斯滤波，服从正态分布，离中心点越近其概率越高，离中心点越远，其概率越低
    其模糊的效果应该比均值滤波要好
    https://blog.csdn.net/Ibelievesunshine/article/details/104881204
    """
    img = cv2.imread(path)
    # 历史图像
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (3, 3)表示高斯滤波器的长和宽都为3，1.3表示滤波器的标准差
    # 其中，(3,3)为滤波器的大小；1.3为滤波器的标准差，
    # 如果标准差这个参数设置为0，则程序会根据滤波器大小自动计算得到标准差。
    out = cv2.GaussianBlur(img, (5, 5), 1)
    new_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    titles = ['origin image', 'gaussina']
    images = [rgbimg, new_out]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def median_blur(path):
    """使用邻域内所有信号的中位数替换中心像素的值，可以在滤除异常值的情况下较好地保留纹理信息。
    该技术会在一定程度上造成图像模糊和失真，滤波窗口变大时会非常明显"""
    img = cv2.imread(path)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lbimg = cv2.medianBlur(img, 3)
    new_out = cv2.cvtColor(lbimg, cv2.COLOR_BGR2RGB)

    titles = ['origin image', 'medianBlur']
    images = [rgbimg, new_out]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    filename = './img/lenaNoise.png'
    # blur_filter_func(filename)
    # box_blur_filter_func(filename)
    # gaussian_blur(filename)
    median_blur(filename)

