# coding: utf-8 
# @时间   : 2021/7/30 8:20 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 单变量线型回归
# @数据   :将使用一个变量实现线性回归，以预测食品卡车的利润。
#         假设你是一家餐馆的首席执行官，正在考虑不同的城市开设一个新的分店。
#         该连锁店已经在各个城市拥有卡车，而且你有来自城市的利润和人口数据。
#         您希望使用这些数据来帮助您选择将哪个城市扩展到下一个城市
# @文件   : lineRegression_2.py.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
1。第一步用图形化的方法来展示数据的构成
"""

def feature_normalization(pd_data):
    """
    将数据进行特征归一化处理。
    原因：当进行多个变量的梯度下降时，如果各个变量的取值范围差别很大，
    即这些特征的尺度相差很大，那么梯度下降时收敛速度可能会非常慢，
    因此为了适当的提高收敛的速度，我们可以通过将所有特征通过缩放到统一尺度来加快收敛速度
    均值归一化（Mean normalization）是比较通用的特征缩放的方法：
    即通过让特征x_{i}尽可能接近0，在除以特征取值范围的最大值减去最小值。
    :param pd_data:
    :return:
    """
    data = (pd_data - pd_data.mean()) / pd_data.std()
    return data
def show_data(path, saveName='./img/line_1.jpg', isShow=True, columns=['Population', 'Profit']):
    """
    图形化展示获取的源数据。用来评估需要用到的算法。
    :param path:
    :return:
    """
    pd_data = pd.read_csv(path, header=None, names=columns)
    if len(columns) == 2:
        pd_data.plot(kind='scatter', x=columns[0], y=columns[-1], figsize=(8, 5))
    elif len(columns)==3:
        pd_data.plot(kind='scatter', x=columns[0], y=columns[-1],figsize=(8, 5))
    else:
        return pd_data
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)
    return pd_data


"""
2。完成线型回归的求解过程
"""


class LR(object):
    def __init__(self, X, y):
        """
        X为x的变量,y为样本数据的实际值
        :param X:
        :param y:
        """
        self.X = X
        self.y = y


    def hTheta_X(self, theta):
        """
        线性回归的激活函数.这里没有考虑偏置项，因为通常说来偏置项为0
        h𝝷(χi) = θ^T * X = θ_0 * x_0 + θ_1 * x_1+θ_2 * x_2 + ..θ_n * x_n
        h𝝷(χi) = θ_0  + θ_1 * x_1
        考虑为什么θ值的变化，会影响到h𝝷(χi)的值。
        假设h𝝷(χi) = θ_0 + θ_1 * x_1的情况下
        设θ_0 = 1.5, θ_1 = 0,则预测值的图为
        h𝝷(χi)
        │
        │
        │
        │─────────────────────────────1.5
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>x
            1   2   3   4   5   6   7
        设θ_0 = 0, θ_1 = 0.5,则此图为
        h𝝷(χi)
        │                2.5➚
        │            2.0➚
        │        1.5➚
        │   0.5➚
        │0➚─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>x
            1   2   3   4   5   6   7
        设θ_0 = 1, θ_1 = 0.5,则预测值的图为
        h𝝷(χi)
        │                3.5➚
        │            3.0➚
        │        2.5➚
        │   1.5➚
        │1➚
        │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>x
        0   1   2   3   4   5   6   7
        此时线性回归的问题就演变为了寻找θ的值。将θ的值，什么时候与训练样本的值最接近，那么这个θ值就最优值。
        :param X:训练样本中的数据，是一个martix向量
        :param theta:θ，即影响线性回归的θ参数。注意此时是一个martix向量
        :return:
        """

        """
        第一次就是
        [                 
          [1,6.1101]      * [0,0] = [0]     
        ]
        第二次就是 
        [                 
          [1,6.1101]      * [0.05839135,0.6532885] = [ 1 * 0.05839135 + 6.1101 * .6532885 ]  = [4.050049413850001]   
        ]                
        """
        hx = self.X * theta.T
        return hx

    def jTheta(self, theta):
        """
        损失函数。在线型回归中我们要解决的是一个最小化的问题, 我们要做的事情就是尽量减少假设输出与房子真实价格之间差的平方
        损失函数为：J𝝷 = 1/2m * ∑(h𝝷(χi) - γi) ** 2
        为什么我们需要使用平方差的公式来做为损失函数？
        假设h𝝷(χi) = θ_0 + θ_1 * x_1的情况下
        设θ_0 = 0, θ_1 = 0.5,则预测值的图为
        y->真实值
        │         y4  ╳
        │
        │      y3  ╳    ℏ2∎
        │
        │   y2  ╳  ℏ1.5∎
        │
        │ y1 ╳   ℏ1∎
        │
        │  ℏ0.5∎
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>x
        0    1   2   3   4   5   6   7
        ╳代表真实值[1,2,3,4],通过h𝝷(χi)的公式，我们可以得到预测值[0.5,1,1.5,2]
        Jθ即 = （ℏi - yi）**2 ≈ 0，能够将预测值最小化靠近真实值。
        所以这里的Jθ = （0.5 - 1)** 2 + (1 - 2) ** 2 + (1.5 - 3)**2 + (2 - 4) ** 2
                   = 0.25 + 1 + 2.25 + 4 = 8 / 2 * 4 = 1
        假设θ_0 = 0, θ_1 = 1，,通过h𝝷(χi)的公式，我们可以得到预测值[1,2,3,4]
        所以Jθ = （1 - 1 ) ** 2 + (2 - 2) ** 2 + (3 - 3) ** 2 + (4 - 4 )** 2
              =0 / 2 * 4 = 0
        假设θ_0 = 0, θ_1 = 1.5,通过h𝝷(χi)的公式，我们可以得到预测值[1.5,3,4.5,6]
        所以Jθ = （1.5 - 1 ) ** 2 + (3 - 2) ** 2 + (4.5 - 3) ** 2 + (6 - 4 )** 2
              = 2.25 + 1 + 2.25 + 8
              =14 / 2 * 4 = 1.75
        通过这种方法不断的改变θ_1的值，则我们可以获取到下图示
        Jθ
        │
        │
        │
        │
        │ ●                 ●
        │  ●              ●
        │   ●           ●Jθ=1.75，当θ=1.5
        │Jθ=1 ●       ●
        │         ●Jθ=0，当θ=1
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>θ_1
        0    0.5   1   1.5   2   2.5
        ---------------------------------------------
        因此要使预测值更加准确，只需要使minJ(θ_1)取得最小值。
        同理，如果θ_0 != 0，则我们只需要使得minJ(θ_0,θ_1)取得最小值
        :param X:
        :param y:
        :param theta:
        :return:
        """
        inner = np.power((self.hTheta_X(theta) - self.y), 2)
        return np.sum(inner) / (2 * len(self.X))

    def gradientDescent(self, theta, alpha, epoch):
        """
        采用梯度下降的方法来求最化小minJ(θ_0,θ_1)的参数值：
        这个时候，就转换为如何求θ_1的最佳值？
        Jθ
        │
        │
        │
        │                     ● ↙
        │ ●                 ● ↙⇡
        │  ●              ● ↙  ⇡
        │   ●           ● ↙⇡   ⇡
        │     ●       ● ↙  ⇡   ⇡
        │        ●         ⇡   ⇡
        │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>θ_1
                        a_k+1  a_k
        求函数Jθ极小值的问题，可以选择任意初始点a0,从a0出发沿着负梯度方向走，可使得Jθ下降最快。
        𝐒(0) = -△J(a_0) s(0)：点a0的搜索方向。
        p_k 为ak 到a_k+1之间的步长
        所以如a_k+1 = a_k + p_k * (- J(a_k)/||Ja_k+1||)
             a_k+1 = a_k - p_k * (- J(a_k)/||Ja_k+1||)
        p_k的步长又称之为学习率。学习率过大，则容易出现过拟合的问题。
        如果放到吴恩达视频中的计算公式，则为
                         ∂
        θ_j = θ_j -α * ──────J(θ_0,...θ_m)
                        ∂θ_j
        因此梯度学习的算法可以概括为：
        我们有一个代价函数J(𝝷0,𝝷1)有两个参数,当然也可能N个,我们需要寻找代价函数的最小值minJ(𝝷0,𝝷1),
        我们刚开始设𝝷0 = rnd,𝝷1 = rnd,然后以一定的频率不断改变(𝝷0,𝝷1),直到达到我们期望的最小值。
        -------------------------------------------------------------------------
          ∂
        ──────J(θ_0,...θ_m) 即求损失函数J𝝷 = 1/2m * ∑(h𝝷(χi) - γi) ** 2的偏导数
         ∂θ_j

         则结果为：
                               m
         temp = θ_j -α * 1/m * ∑ (h𝝷(χi) - γi) * χi
                               i=1
         θ_j = temp

        :param alpha: 学习率
        :param epoch: 学习次数
        :return:
        """
        temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵，用来存储计算出来的theta值
        parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
        cost = np.zeros(epoch)  # 初始化一个ndarray，用来存储每次epoch的cost
        m = self.X.shape[0]  # 样本数量m
        for i in range(epoch):
            # 利用向量化一步求解

            # f1 = self.hTheta_X(theta)
            # f2 = f1 - self.y
            # f3 = f2.T#转置是为了获取theta
            # f4 = (alpha /m ) * f3
            # f4_1 = (alpha / m) * f2
            # f5 = f4 * self.X
            # f5_1 = f4_1.T * self.X
            # f6 = theta - f5

            temp = theta - (alpha / m) * (self.hTheta_X(theta) - self.y).T * self.X
            # 以下是不用Vectorization求解梯度下降
            #         error = (self.X * theta.T) - y  # (97, 1)
            #         for j in range(parameters):
            #             term = np.multiply(error, X[:,j])  # (97, 1)
            #             temp[0,j] = theta[0,j] - ((alpha / m) * np.sum(term))  # (1,1)
            theta = temp
            cost[i] = self.jTheta(theta)
        return theta, cost


"""
3。验证第2步中求出的θ是否能够拟合数据。
"""


def verfiy_theta(data, final_theta, isShow=True, saveName='./img/line_lr_1.jpg'):
    """
    验证得到的θ是否能够拟合原始数据
    :param data:
    :param final_theta:
    :param isShow:
    :param saveName:
    :return:
    """
    x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
    f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data['Population'], data.Profit, label='Traning Data')
    ax.legend(loc=2)  # 2表示在左上角
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)


def verfiy_epoch(epoch, cost, isShow=True, saveName='./img/line_lr_epoch_1.jpg'):
    """
    原文中，有一个观点，迭代次数增加到一定的次数后，其损失函数并不会快速的减少。这个函数就是验证这一点的。
    :param epoch:
    :param cost:
    :param isShow:
    :param saveName:
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)

def main():
    pd_data = show_data('./data/ex1data1.txt', isShow=False)
    # pd_data = feature_normalization(pd_data)
    # 增加一列值用来设置偏置项
    pd_data.insert(0, 'Ones', 1)
    # 随机选择60%的数据用来做训练集的内容。
    pd_data = pd_data.sample(frac=0.6)
    cols = pd_data.shape[1]  # 列数
    X = pd_data.iloc[:, 0:cols - 1]  # 取前cols-1列，即输入向量X
    y = pd_data.iloc[:, cols - 1:cols]  # 取最后一列，即目标向量y
    # 设置学习率
    alpha = 0.003
    # 设置迭代次数
    epoch = 1100
    # 转换为向量
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # 设置初始theta
    theta = np.matrix([0, 0])
    l = LR(X, y)
    theta, cost = l.gradientDescent(theta, alpha, epoch)
    print("预测出来的变量值{},cost={}".format(theta, cost))
    # verfiy_epoch(epoch, cost, isShow=True)
    # verfiy_theta(pd_data, theta, isShow=True)

if __name__ == "__main__":
    main()





