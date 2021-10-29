# coding: utf-8 
# @时间   : 2021/8/2 8:32 下午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   : 逻辑回归
# @数据   : 假设你是一所大学的行政管理人员，你想根据两门考试的结果，来决定每个申请人是否被录取
# @文件   : LogisticRegression_1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(isShow=True, saveName='./img/logistic_1.jpg'):
    """
    读取原数据
    :return:
    """
    data = pd.read_csv('./data/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    # 图型化展示其数据
    positive = data[data.admitted == 1]  # 1
    negetive = data[data.admitted == 0]  # 0
    fig, ax = plt.subplots(figsize=(6, 5))
    # 散列图
    ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
    ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
    # 设置图例显示在图的上方
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12), ncol=3)
    # 设置横纵坐标名
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)
    return data


class LGR():
    """
    1, 为什么要用逻辑回归？
       从图例中可以看出来似乎用线型回归也可以用一条线进行区分，但是由于h𝝷(χi) = θ^T * X = θ_0 * x_0 + θ_1 * x_1+θ_2 * x_2 + ..θ_n * x_n
    的值，可能是大于1或者小于0的，而我们源数据的值只能是在[0,1]之间的范围，且>0.5为1，<0.5为0，所以是不适用的。
    2, 为什么sigmoid函数可以用来进行逻辑回归？
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sigmoid(self, z):
        """
        逻辑回归的激活函数，其中z = θ^T * X = θ_0 * x_0 + θ_1 * x_1+θ_2 * x_2 + ..θ_n * x_n
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(- z))

    def hLGRTheta(self, theta):
        """
        逻辑回归的损失函数为
                        {
                            -log(hθ(x)), if y == 1
        Cost(hθ(x),y) =     -log(1 - hθ(x)), if y == 0
                        }
        verfiy_log()函数表明：
        如果我们预测的h𝝷(x) = 0 ,但是实际值y = 1,那么cost的损失会趋于无穷. 如果h𝝷(x)=1,并且y=1,那么其cost会趋于0.
        所以这个函数非常适合用来做逻辑回归的损失函数

        利用极大似然法的算法，则计算的公式合二为一表示为
                                 m
        Cost(hθ(x),y) = 1/m * [ ∑ -y^i * log(hθ(x^i)) - (1 - y^i) * log(1 - hθ(x^i)]
                                i=1

        又等于                    m
        Cost(hθ(x),y) = -1/m * [ ∑ y^i * log(hθ(x^i)) + (1 - y^i) * log(1 - hθ(x^i)]
                                i=1

        所以Jθ = Cost(hθ(x),y)
        :param theta:
        :return:
        """
        first = (-self.y) * np.log(self.sigmoid(self.X @ theta))
        second = (1 - self.y) * np.log(1 - self.sigmoid(self.X @ theta))
        # 1/m * (-A - B) = -1/m (A + B)
        result = np.mean(first - second)
        return result

    def gradientDescent(self, alpha, epoch, theta):
        """
        梯度下降
        :return:
        """
        temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵，用来存储计算出来的theta值
        cost = np.zeros(epoch)
        m = self.X.shape[0]
        for i in range(epoch):
            temp = theta - alpha  * ((self.sigmoid(self.X @ theta) - self.y) @ self.X)
            theta = temp
            cost[i] = self.hLGRTheta(theta)
        return theta, cost

    def predict(self, theta, X):
        """
        当hθ(x)>0.5时，则预测为1;否则为0;
        :param theta:
        :return:
        """
        probability = self.sigmoid(X @ theta)
        return [1 if x >= 0.5 else 0 for x in probability]

    def veriy_LRG_correct(self, theta, X, y):
        """
        在验证集中验证其准确率是多少
        :param theta:
        :param X:
        :return:
        """
        predictions = self.predict(theta, X)
        correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
        accuracy = sum(correct) / len(X)
        return accuracy


def verfiy_sigmoid(isShow=True, saveName='./img/logistic_verfiy_sigmoid.jpg'):
    """
    验证为什么用sigmoid函数，从图中我们可以看出这个函数的图像变为,当z>0时,y为[0.5,1]
    当z<0时,y为[0,0.5],所以0.5为分割后一条曲线,z无穷大时,趋近于1,反之则趋近于0.
    很适合用来做逻辑回归的激活函数
    :return:
    """
    t = LGR(0, 0)
    x1 = np.arange(-10, 10, 0.1)
    plt.plot(x1, t.sigmoid(x1), c='r')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)


def verfiy_log(isShow=True, saveName='./img/logistic_verfiy_z.jpg'):
    x1 = np.arange(0.01, 1, 0.01)
    plt.plot(x1, -np.log(x1), c='r', )
    plt.plot(x1, np.log(1 - x1), c='b')
    # 绘制网格
    plt.grid(alpha=0.4, linestyle=':')
    if isShow:
        plt.show()
    else:
        plt.savefig(saveName)

def verfiy_result(final_theta):
    """
    验证得到的final_theta，在图形中的区分是怎样的
    :param final_theta:
    :return:
    """
    x1 = np.arange(130, step=0.1)
    x2 = -(final_theta[0] + x1 * final_theta[1]) / final_theta[2]
    data = pd.read_csv('./data/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    # 图型化展示其数据
    positive = data[data.admitted == 1]  # 1
    negetive = data[data.admitted == 0]  # 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
    ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.plot(x1, x2)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 130)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary')
    plt.show()


if __name__ == "__main__":
    # data = read_csv(isShow=False)
    #################################################################################
    # 验证为什么是sigmoid这个函数
    verfiy_sigmoid()

    #################################################################################
    # 验证为什么损失函数要用log()
    # verfiy_log()
    #################################################################################
    # add a ones column - this makes the matrix multiplication work out easier
    # if 'Ones' not in data.columns:
    #     data.insert(0, 'Ones', 1)
    # # set X (training data) and y (target variable)
    # data = data.sample(frac=0.6)
    # X = data.iloc[:, :-1].values  # Convert the frame to its Numpy-array representation.
    # y = data.iloc[:, -1].values  # Return is NOT a Numpy-matrix, rather, a Numpy-array.
    # #################################################################################
    # theta = np.zeros(X.shape[1])
    # l = LGR(X, y)
    # final_theta, cost = l.gradientDescent(0.00001, 200000, theta)
    # print(l.hLGRTheta(final_theta),final_theta)
    # # #################################################################################
    # # # 将数据用来做验证
    # data = data.sample(frac=0.2)
    # X = data.iloc[:, :-1].values
    # y = data.iloc[:, -1].values
    # print(l.veriy_LRG_correct(final_theta, X, y))
    # #################################################################################
    # 图形化验证
    # verfiy_result(np.array([-7.45017822 , 0.06550395 , 0.05898701]))


