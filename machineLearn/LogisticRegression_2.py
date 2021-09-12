# coding: utf-8 
# @时间   : 2021/8/3 6:01 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 逻辑回归增加正则化,通过加入正则项提升逻辑回归算法，有助于避免过拟合的问题。
#          过拟合即训练出来的h𝝷值能够将所有的样本进行区分,所得到的曲线极度扭曲此时J𝝷值趋近于0,无法很好地泛化到新的样本数据.
#          当我们样本数据的特征非常多,但训练数据非常少时,就会出现过拟合的情况
#           1. 人工选择最相关的变量特征,但是有可能舍弃一些相关的特征.
#           2. 正则化来进行实现. 我们保留所有特征,但减少样本的数量级𝝷j的大小.
# @数据   : 设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果
# @文件   : LogisticRegression_2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LogisticRegression_1 import LGR


def read_csv(isShow=True, saveName='./img/logistic_2.jpg'):
    data = pd.read_csv('./data/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)
    return data


def add_polynomial(x1, x2, power):
    """
    我们知道z = θ^T * X 。假设只有2个变量，则 θ_0 + θ_1 * x1 + θ_2 * x2
    令θ_0 = -3, θ_1 = 1,θ_2 = 1
    如，则P(y=1|x:θ)为1
    则, -3 + x1 + x2 >0，则x1 + x2 > 3;
    如，P(y=0|x:θ)为0，则   x1 + x2 < 3；
    则这一条线，叫决策边界线。
    需要注意，决策边界线可能是个曲线。通过多项式的增加，可以使决策边界线变成一个多边形的线，比如
    θ_0 + θ_1 * x1 + θ_2 * x2 + θ_3 * x1^2 + θ_4 * x1 * x2 + θ_5 * x2^2 + ...
    多项式的增加维度，可能会使得曲线极度扭曲。
    比如：θ_0 + θ_1 * x1^2 + θ_2 * x2^2,令θ_0 = -1,θ_1=0,θ_2 =0,则
    -1 + x1^2 + x2^2  >0，则y = 1
    -1 + x1^2 + x2^2  <0, 则y = 0
    则x1^2 + x2^2 > 1; x1^2 + x2^2 < 1，则此时就是一个圆了

    高维特征向量上训练的logistic回归分类器将会有一个更复杂的决策边界，当我们在二维图中绘制时，会出现非线性。
    虽然特征映射允许我们构建一个更有表现力的分类器，但它也更容易过拟合.
    :return:
    """
    data = {}
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return pd.DataFrame(data)


class Regularization_LRG(LGR):
    """
    为了解决升维带来的过拟合问题，增加正则化来进行数据的拟合
    """

    def __init__(self, X, y):
        super().__init__(X, y)

    def hLGRReTheta(self, theta, C):
        """
        针对损失函数进行正则化的惩罚
                                 m
        Cost(hθ(x),y) = -1/m * [ ∑ y^i * log(hθ(x^i)) + (1 - y^i) * log(1 - hθ(x^i)]
                                i=1
                                    m
        Jθ = Cost(hθ(x),y) + l/2m * ∑ θ_j^2
                                    j=1
        如果设置正则化参数值过大,会导致出现𝝷_1至𝝷_n的值趋近于0才能使J𝝷的值最小化,则此时就出现了欠拟合的情况,
        因为此时的是𝝷0的值,即一条平行线.
        :param theta:
        :param l:
        :return:
        """
        # 不对第0个θ进行惩罚
        _theta = theta[1:]
        reg = (C / (2 * len(self.X))) * (_theta @ _theta)  # _theta@_theta == inner product
        return self.hLGRTheta(theta) + reg

    def ReGradientDescent(self, alpha, epoch, theta, C):
        """
        加入正则化的梯度下降
        :param alpha:
        :param epoch:
        :param theta:
        :return:
        """
        # reg = (C / len(X)) * theta
        # reg[0] = 0
        # theta,cost = self.gradientDescent(alpha,epoch,theta)
        # theta = theta + reg
        # return theta
        temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵，用来存储计算出来的theta值
        cost = np.zeros(epoch)
        m = self.X.shape[0]
        for i in range(epoch):
            temp = theta * (1 - (alpha * (C / m))) - (alpha * (1 / m)) * ((self.sigmoid(self.X @ theta) - self.y) @ self.X)
            theta = temp
            cost[i] = self.hLGRReTheta(theta,C)
        return theta, cost


def verify_reg(final_theta):
    """
    决策边界为一个椭圆
    :param final_theta:
    :return:
    """
    x = np.linspace(-1, 1.5, 250)
    xx, yy = np.meshgrid(x, x)

    z = add_polynomial(xx.ravel(), yy.ravel(), 6).values
    z = z @ final_theta
    z = z.reshape(xx.shape)

    data = pd.read_csv('./data/ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')

    plt.contour(xx, yy, z, 0)
    plt.ylim(-.8, 1.2)
    plt.show()


if __name__ == "__main__":
    data = read_csv(isShow=False)
    x1 = data['Test 1'].values
    x2 = data['Test 2'].values
    y = data['Accepted'].values
    # 对源数据进行升维
    data = add_polynomial(x1, x2,6)
    if 'Ones' not in data.columns:
        data.insert(0, 'Ones', 1)
    X = data.iloc[:, :-1].values

    # theta = np.zeros(X.shape[1])
    # l = LGR(X, y)
    # final_theta, cost = l.gradientDescent(0.0000001, 200000, theta)
    # print(l.hLGRTheta(final_theta), final_theta)

    theta = np.zeros(X.shape[1])
    l = Regularization_LRG(X,y)
    a = 0.01
    final_theta, cost = l.ReGradientDescent(0.003,20000,theta,a)

    # final_theta = np.array([0.57761135, 0.47056293, 1.09213933, -0.93555548, -0.15107417,
    #                         -0.96567576, -0.49622178, -0.87226365, 0.5986215, -0.47857791,
    #                         -0.19652206, -0.10212812, -0.1513566, -0.03407832, -1.868297,
    #                         -0.25062387, -0.49045048, -0.20293012, -0.26033467, 0.02385201,
    #                         -0.0290203, -0.0543879, 0.01131411, -1.39767636, -0.16559351,
    #                         -0.24745221, -0.29518657, 0.00854288])
    print(min(cost))
    print(l.hLGRReTheta(final_theta,a), final_theta)
    # verify_reg(final_theta)
