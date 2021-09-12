# coding: utf-8
# @时间   : 2021/8/15 2:53 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 反向传播
# @文件   : Backpropagation_1.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告
from sklearn.preprocessing import OneHotEncoder

"""
L->网络层数；S_i -> 神经元的个数；S_l -> 输出层的神经元个数；
将神经网络的分类定义为两种情况:二类分类和多类分类
二类分类:S_l = 0 , y = 0 或才1表示那一类；
K类分类：S_l = k, y_i = 1表示分类到那个i类；

逻辑回归的代价函数：
                                 m
        Cost(hθ(x),y) = -1/m * [ ∑ y^i * log(hθ(x^i)) + (1 - y^i) * log(1 - hθ(x^i)]
                                i=1
                                    m
        Jθ = Cost(hθ(x),y) + l/2m * ∑ θ_j^2
                                    j=1
因为此时，hθ(x^i) 可能是 h𝜃(𝑥) ∈ R^k,并且(h𝜃(𝑥))^i = 𝑖^𝑡h 输出，所以此时代价函数为：

                 1    m    k                                                                   l   L-1 S_l s_l+1
        J𝛩 = - ———— [ ∑    ∑     y_k^i * log(h𝛩(x^i))_k + (1 - y_k^i) * log(1 - h𝛩(x^i))_k] + ———— ∑   ∑   ∑     𝛩_ij^l ** 2
                m     i=1  k=1                                                                 2m  l=1 i=1 j=1

对于每一行特征，我们都会给出𝐾个预测，基本上我们可以利用循环，对每一行特征都预测𝐾个不同结果，然后在利用循环在𝐾个预测中选择可能性最高的一个，
将其与𝑦中的实际数据进行比较
                    ∂
反向传传播的目的是求 —————————— J(𝛩)的值，即从l_last到l2中的最小误差。首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层
                  ∂𝛩_ij^l
-----------------------------------------------------------------------------------------------------------------------
假设 k = 4,S_l = 4, l = 4，则前向传播为：
a1 = x
z2 = a1 * 𝛩1
a2 = g(z2)
-----------
z3 = a2 * 𝛩2
a3 = g(z3)
-----------
z4 = a3 * 𝛩3
a4 = g(z4) = h𝛩(x)
我们从最后一层的误差开始计算，误差是激活单元的预测,a_k^4 - y^k之间的误差，表示为𝛿^4 = a_k^4 - y
利用这个误差值来计算前一层的误差:
𝛿(3) = 𝛩3^T 𝛿^4  * g'(z3), g'(z3)是S型函数的导数，𝑔′(𝑧(3)) = 𝑎(3) ∗ (1 − 𝑎(3))。 𝛩3^T 𝛿^4权重导致的误差的和。

𝛿(2) = 𝛩2^T 𝛿^3  * g'(z2)

因为第一层是输入变量，不存在误差。我们有了所有的误差的表达式后，便可以计算代 价函数的偏导数了，假设𝜆 = 0，即我们不做任何正则化处理时有：
    ∂
 —————————— J(𝛩) = a_j^l * 𝛿_i^l + 1
  ∂𝛩_ij^l
𝑙 代表目前所计算的是第几层。
𝑗 代表目前计算层中的激活单元的下标，也将是下一层的第𝑗个输入变量的下标。
𝑖 代表下一层中误差单元的下标，是受到权重矩阵中第𝑖行影响的下一层中的误差单元的下标。
-----------------------------------------------------------------------------------------------------------------------
如果我们考虑正则化处理，并且我们的训练集是一个特征矩阵而非向量。在上面的特殊情况中，我们需要计算每一层的误差单元来计算代价函数的偏导数。
在更为一般的情况中， 我们同样需要计算每一层的误差单元，但是我们需要为整个训练集计算误差单元，此时的误差单元也是一个矩阵.
我们用𝛥_ij^(𝑙)来表示这个误差矩阵。第 𝑙 层的第 𝑖 个激活单元受到第𝑗个参数影响而导致的误差。
则我们的算法表示为：
for x in range(m):
    set a^i = x^i
    𝛿^l = a^l - y^i
    𝛥_ij^(𝑙) = 𝛥_ij^(𝑙) +  a_j^l * 𝛿_i^l + 1
即首先用正向传播方法计算出每一层的激活单元，利用训练集的结果与神经网络预测的 结果求出最后一层的误差，然后利用该误差运用反向传播法计算出直至第二层的所
有误差。
在求出了𝛥_ij^(𝑙)之后，我们便可以计算代价函数的偏导数了，计算方法如下:

D_ij^(𝑙) := 1/m 𝛥_ij^(𝑙) + ⋋ 𝛩_ij^(l), if j != 0
D_ij^(𝑙) := 1/m 𝛥_ij^(𝑙),              if j  = 0
-----------------------------------------------------------------------------------------------------------------------
梯度的数值检验：
当我们对一个较为复杂的模型(例如神经网络)使用梯度下降算法时，可能会存在一些 不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是
最优解。
对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后 计算两个点的平均值用以估计梯度。即对于某个特定的 𝜃，
我们计算出在 𝜃-𝜀 处和 𝜃+𝜀 的 代价值(𝜀是一个非常小的值，通常选取 0.001)，然后求两个代价的平均，用以估计在 𝜃 处的代价值
-----------------------------------------------------------------------------------------------------------------------
随机初始化：
任何优化算法都需要一些初始的参数。到目前为止我们都是初始所有参数为 0，这样的 初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。
如果我们令所有 的初始参数都为 0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我 们初始所有的参数都为一个非 0 的数，结果也是一样的。
我们通常初始参数为正负𝜀之间的随机值

小结一下使用神经网络时的步骤:
网络结构:第一件要做的事是选择网络结构，即决定选择多少层以及决定每层分别有多 少个单元。
第一层的单元数即我们训练集的特征数量。 最后一层的单元数是我们训练集的结果的类的数量。
如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。
我们真正要决定的是隐藏层的层数和每个中间层的单元数。
训练神经网络:
1. 参数的随机初始化
2. 利用正向传播方法计算所有的h𝜃(𝑥)
3. 编写计算代价函数 𝐽 的代码
4. 利用反向传播方法计算所有偏导数
5. 利用数值检验方法检验这些偏导数
6. 使用优化算法来最小化代价函数
"""


def load_mat():
    '''读取数据'''
    data = loadmat('./data/ex4data1.mat')  # return a dict
    X = data['X']
    y = data['y'].flatten()

    return X, y


def plot_100_images(X):
    """随机画100个数字"""
    index = np.random.choice(range(5000), 100)
    images = X[index]
    fig, ax_array = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(images[r * 10 + c].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def expand_y(y):
    """
    把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1
    :param y:
    :return:
    """
    result = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        result.append(y_array)
    '''
    # 或者用sklearn中OneHotEncoder函数
    encoder =  OneHotEncoder(sparse=False)  # return a array instead of matrix
    y_onehot = encoder.fit_transform(y.reshape(-1,1))
    return y_onehot
    '''
    return np.array(result)


def load_weight():
    """已训练好的权重"""
    data = loadmat('./data/ex4weights.mat')
    return data['Theta1'], data['Theta2']


def serialize(a, b):
    '''展开参数'''
    return np.r_[a.flatten(), b.flatten()]


def sigmoid(z):
    """激活函数"""
    return 1 / (1 + np.exp(- z))


def deserialize(seq):
    '''提取参数'''
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def feed_forward(theta, X, ):
    '''得到每层的输入和输出'''
    t1, t2 = deserialize(theta)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = X
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3


def cost(theta, X, y):
    """神经网络的代价函数"""
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    '''
         # or just use verctorization
         J = - y * np.log(h) - (1 - y) * np.log(1 - h)
         return J.sum() / len(X)
    '''
    return J


def regularized_cost(theta, X, y, l=1):
    '''正则化时忽略每层的偏置项，也就是参数矩阵的第一列'''
    t1, t2 = deserialize(theta)
    reg = np.sum(t1[:, 1:] ** 2) + np.sum(t2[:, 1:] ** 2)  # or use np.power(a, 2)
    return l / (2 * len(X)) * reg + cost(theta, X, y)


def sigmoid_gradient(z):
    """激活函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


def random_init(size):
    '''从服从的均匀分布的范围中随机返回size大小的值'''
    return np.random.uniform(-0.12, 0.12, size)


def gradient(theta, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    # 输出层预测值与实际值之间的误差
    d3 = h - y  # (5000, 10)
    # 第二层的预测值与实际值之间的误差
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    # 第2层的偏导数
    D2 = d3.T @ a2  # (10, 26)
    # 第1层的偏导数
    D1 = d2.T @ a1  # (25, 401)
    # 计算代价函数的偏导数
    D = (1 / len(X)) * serialize(D1, D2)  # (10285,)
    return D


def regularized_gradient(theta, X, y, l=1):
    """不惩罚偏置单元的参数，正则化神经网络"""
    # a1, z2, a2, z3, h = feed_forward(theta, X)
    D1, D2 = deserialize(gradient(theta, X, y))
    t1[:, 0] = 0
    t2[:, 0] = 0

    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2

    return serialize(reg_D1, reg_D2)


def nn_training(X, y):
    """Learning parameters using fmincg 优化参数"""
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res


def accuracy(res, X, y):
    _, _, _, _, h = feed_forward(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    X, y = load_mat()
    X = np.insert(X, 0, 1, axis=1)
    y = expand_y(y)
    t1, t2 = load_weight()
    theta = serialize(t1, t2)
    # 前向传播
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = cost(theta, X, y)
    Jy = regularized_cost(theta, X, y)
    # 反向传播
    D = gradient(theta, X, y)
    # 正则化惩罚
    rD = regularized_gradient(theta, X, y)
    # 参数优化
    res = nn_training(X, y)
    # accuracy(res, X, y)
    pass
    # plot_100_images(X)
