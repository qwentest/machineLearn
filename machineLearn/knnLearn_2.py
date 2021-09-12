import math
import random
import  matplotlib.pyplot as plt


class KNN():
    def __init__(self, dataSet):
        self.dataSet = dataSet
        self.disData = None

    def euclidean_distance(self, labels):
        """欧式距离的计算"""
        distance = []
        for k, v in self.dataSet.items():
            pow_sum = 0
            for i in range(len(labels)):
                pow_sum += pow(v[i] - labels[i], 2)
            distance.append((k, math.sqrt(pow_sum)))
        self.disData = sorted(distance, key=lambda x: x[1])
        return self.disData

    def knn_guess(self, k):
        #自动地去源数据中获取所有的电影类型
        labels = {name: 0 for name in set([v[-1] for key, v in self.dataSet.items()])}
        for name, dis in self.disData[:k]:
            labels[self.dataSet.get(name)[3]] += 1
        labels = sorted(labels.items(), key=lambda l: l[1], reverse=True)
        return labels[0][0], round(labels[0][1] / sum([v for l, v in labels]), 2) * 100

class ZKNN(KNN):
    def __init__(self,dataSet):
        KNN.__init__(self,dataSet)

    def getInitData(self):
        data = [(r[-1],r[0:3]) for r in self.dataSet.values()]
        random.shuffle(data)

        type_data = []
        init_data = []
        #随机得到初始训练数据，在这里每种类型选择一种A
        i = 0
        while i<len(data):
            rdata = random.choices(data)
            if rdata[0][0] not in type_data:
                type_data.append(rdata[0][0])
                init_data.append((rdata[0][0],rdata[0][1]))
                # init_data[rdata[0][0]] = rdata[0][1]
            i += 1
        #得到B组数据.B组数据中应该排除A种数据。
        train_data =  [ i for i in data if i not in init_data ]

        #重构为父类的格式
        d = {}
        i = 0
        for k,v in train_data:
            d[i] = v + [k]
            i += 1
        return init_data,d
    def trainKnn(self,kk=5):
        """将初始的数据参与knn的运算"""
        trainData = self.getInitData()
        #使用B中的数据去训练
        self.dataSet = trainData[1]

        for k,v in self.dataSet.items():
            self.euclidean_distance(v[0:3])
            k = kk
            result = self.knn_guess(k)
            if v[-1] != result[0]:
                print('error: 原有数据类型为|{},猜出来的为{}|'.format(v,result))
                trainData[0].append((v[-1],v[0:3]))

            elif result[1] < 75.0:
                pass
                # print('虽然两者一致，但是概率低于75%，原来|为{},猜出来的概率为{}|'.format(v[-1],result))
            else:
                pass
                # print('两者一致，且概率>75%，原来为|{},猜出来的概率为{}|'.format(v, result))
        return trainData[0]


if __name__ == "__main__":
    movie_data = {"宝贝当家": [45, 2, 9, "喜剧片"],
                  "美人鱼": [21, 17, 5, "喜剧片"],
                  "澳门风云3": [54, 9, 11, "喜剧片"],
                  "功夫熊猫3": [39, 0, 31, "喜剧片"],
                  "谍影重重": [5, 2, 57, "动作片"],
                  "叶问3": [3, 2, 65, "动作片"],
                  "伦敦陷落": [2, 3, 55, "动作片"],
                  "我的特工爷爷": [6, 4, 21, "动作片"],
                  "奔爱": [7, 46, 4, "爱情片"],
                  "夜孔雀": [9, 39, 8, "爱情片"],
                  "代理情人": [9, 38, 2, "爱情片"],
                  "新步步惊心": [8, 34, 17, "爱情片"]}


    #
    #
    #训练10次，得到那些可能出现异常的数据。
    # r = []
    # kk = 4
    # for x in range(10):
    #     print('第%d次训练=========开始'%x)
    #     zk = ZKNN(movie_data)
    #     result = zk.trainKnn(kk=kk)
    #     r += result
    #     print('第%d次训练=========结束' % x)
    #
    # #去重复，得到具有最大特征的样本数据。
    # newList = []
    # for d in r:
    #     if d not in newList:
    #         newList.append(d)
    # print('用于训练的数据{}\n最后的数据的长度为{}'.format(newList,len(newList)))
    # #组合成字典的方法，然后去猜测指定的内容。
    # d = {}
    # i = 0
    # for k, v in newList:
    #     d[i] = v + [k]
    #     i += 1
    # k1 = KNN(d)
    # k1.euclidean_distance([23, 3, 17])
    # result = k1.knn_guess(kk)
    # print(result)

    def guess_k(lenged=0.4):
        """利用已有数据来对K值进行验证。"""
        t = KNN(movie_data)
        data_keys = movie_data.keys()
        data_values = [(d[-1], d[:3]) for d in movie_data.values()]
        # random.shuffle(data_values)
        # 通常说来取源数据的40%。
        data_values = data_values[0:int(len(data_keys) * lenged)]
        max_k = len(data_keys)
        verfiy_k = []
        verfiy_error = []
        verfiy_wrong = []
        for k in range(1, max_k):
            error_count = 0
            guss_count = 0
            for l, d in data_values:
                t.euclidean_distance(d)
                result = t.knn_guess(k)
                if l != result[0]:
                    error_count += 1
                guss_count += 1
            verfiy_k.append(k)
            verfiy_error.append(error_count)
            verfiy_wrong.append(round(error_count / guss_count,2))
        return verfiy_k, verfiy_error, verfiy_wrong


    # r = guess_error()
    # print(r)
    # 通常只有源数据中的一部分， 这里数据比较少，所以使用全部
    error = guess_k(1)
    # #画出K的取值错误趋势图
    fig = plt.figure(num=12)
    ax1 = fig.add_subplot(111)
    ax1.plot(range(1, 12), error[-1])
    ax1.set_xlabel('k')
    ax1.set_ylabel('Error Rate')
    for a, b in zip(range(1, 12), error[-1]):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
    plt.show()

    # t = KNN(movie_data)
    # t.euclidean_distance([23, 3, 17])
    # print(t.knn_guess(k=4))

    # print(math.sqrt(12) / 2)
