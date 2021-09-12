# coding: utf-8 
# @时间   : 2021/8/1 5:34 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 线性回归使用第三方库来完成
# @文件   : lineRegression_3.py
from sklearn import linear_model
import machineLearn.lineRegression_1 as line
import machineLearn.lineRegression_2 as line2

if __name__ == "__main__":
    pd_data = line.show_data('./data/ex1data2.txt',saveName='./img/line_2.jpg',
                             columns=['Size', 'Bedrooms', 'Price'],isShow=False)
    pd_data = line2.feature_normalization(pd_data)
    pd_data.insert(0, 'Ones', 1)
    # 随机选择60%的数据用来做训练集的内容。
    pd_data = pd_data.sample(frac=0.6)
    cols = pd_data.shape[1]  # 列数
    X = pd_data.iloc[:, 0:cols - 1]  # 取前cols-1列，即输入向量X
    y = pd_data.iloc[:, cols - 1:cols]  # 取最后一列，即目标向量y
    model = linear_model.LinearRegression()
    model.fit(X, y)
