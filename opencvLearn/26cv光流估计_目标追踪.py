# coding: utf-8 
# @时间   : 2021/9/7 7:41 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 26cv光流估计_目标追踪.py
"""
https://blog.csdn.net/qq_42239685/article/details/104755100
光流是空间运动物体在观测成象平面上的像素运动的'瞬时速度'。
亮度恒定：同一点随着时间的变化，其亮度不会发生改变。一帧与一帧之间不会有大的变化。
小运动：随着时间的变化不会引起位置的剧烈变化
空间一致：邻近像素点，且邻近点的速度一致。
"""
import numpy as np
import cv2

cap = cv2.VideoCapture('./img/test_move.mov')

# 角点检测所需参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7)

# lucas kanade参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2)

# 随机颜色条
color = np.random.randint(0, 255, (100, 3))

# 拿到第一帧图像
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 返回所有检测特征点，需要输入图像，角点最大数量（效率），品质因子（特征值越大的越好，来筛选）
# 距离相当于这区间有比这个角点强的，就不要这个弱的了
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个mask
mask = np.zeros_like(old_frame)

while (True):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 需要传入前一帧和当前图像以及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1表示
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()