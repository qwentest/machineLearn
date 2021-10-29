# coding: utf-8 
# @时间   : 2021/10/27 3:10 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : kmeansImage.py
# from pylab import imread,imshow,figure,show,subplots
# from cv2 import imread
import cv2
from numpy import reshape,flipud
from sklearn.cluster import KMeans
from copy import deepcopy

img = cv2.imread('./img/girl.png')
pixel = reshape(img, (img.shape[0] * img.shape[1],3))
pixel_new = deepcopy(pixel)
model = KMeans(n_clusters=10)
labels = model.fit_predict(pixel)
palette = model.cluster_centers_
for i in range(len(pixel)):
    pixel_new[i,:] = palette[labels[i]]
result = reshape(pixel_new,(img.shape[0],img.shape[1],3))
cv2.imshow('r', result)
cv2.waitKey(0)
cv2.destroyAllWindows()





