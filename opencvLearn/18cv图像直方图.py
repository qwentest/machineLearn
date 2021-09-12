# coding: utf-8 
# @时间   : 2021/8/30 2:31 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 18cv图像直方图.py
"""
https://blog.csdn.net/zong596568821xp/article/details/116003681
图像直方图是用一表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数。可以借助观察该直方图了解需要如何调整亮度分布的直方图。
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 在服务端调试不显示
matplotlib.use('Agg')


# 计算灰度图的直方图
def calchist_for_gray(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    plt.plot(hist, color="r")
    plt.savefig("result_gray.jpg")
    # plt.show()


# 计算彩色图的直方图
def calchist_for_rgb(imgname):
    img = cv2.imread(imgname)
    histb = cv2.calcHist([img], [0], None, [256], [0, 255])
    histg = cv2.calcHist([img], [1], None, [256], [0, 255])
    histr = cv2.calcHist([img], [2], None, [256], [0, 255])

    plt.plot(histb, color="b")
    plt.plot(histg, color="g")
    plt.plot(histr, color="r")
    plt.savefig("result_rgba.jpg")


# 计算掩码的直方图
def calchist_for_mask(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)

    mask = np.zeros(img.shape, np.uint8)
    mask[200:400, 200:400] = 255

    histMI = cv2.calcHist([img], [0], mask, [256], [0, 255])
    histImage = cv2.calcHist([img], [0], None, [256], [0, 255])

    plt.plot(histMI, color="r")
    plt.savefig("result_mask.jpg")
    # plt.show()


def get_mask(imgname):
    image = cv2.imread(imgname, 0)
    mask = np.zeros(image.shape, np.uint8)
    mask[200:400, 200:400] = 255
    mi = cv2.bitwise_and(image, mask)
    cv2.imwrite("mi.jpg", mi)


def get_equalizehist_img(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)

    plt.subplot(221)
    plt.imshow(img, plt.cm.gray)
    plt.axis('off')

    plt.subplot(222)
    plt.hist(img.ravel(), 256)

    plt.subplot(223)
    plt.imshow(equ, plt.cm.gray)
    plt.axis('off')

    plt.subplot(224)
    plt.hist(equ.ravel(), 256)
    plt.savefig("result2.jpg")


if __name__ == "__main__":
    test_img = "./img/girl.png"
    # calchist_for_rgb(test_img)
    calchist_for_gray(test_img)
    # calchist_for_mask(test_img)
    # get_mask(test_img)
    # get_equalizehist_img(test_img)

