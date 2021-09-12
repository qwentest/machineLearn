# coding: utf-8 
# @时间   : 2021/9/10 10:55 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 19_模型的训练保存和加载.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential, models, losses, metrics
import matplotlib.pyplot as plt

# layers类是网络层的父类，定义了网络层的一些常见功能，如添加权值、管理权值列表等。
# models类是网络类的父类，除了具有layer类的功能，还添加了保存模型、加载模型、训练模型、测试模型等功能。
# Sequential类是models类的子类。
# model -> layers -> Conv2D
#       -> Sequential
#
if __name__ == "__main__":
    # 训练集
    train_db = tf.random.normal([4, 28 * 28])
    # 验证集
    val_db = tf.random.normal([4, 28 * 28])
    network = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    network.build(input_shape=(4, 28 * 28))
    network.summary()
    # compile()指定网络使用的优化器对象、损失函数类型，评价指标等设定。
    # Adam()，学习率为0.01且采用交叉熵的损失函数，包含softmax函数的使用。
    network.compile(optimizer=optimizers.Adam(lr=0.01),
                    loss=losses.CategoricalCrossentropy(from_logits=True),  # 预测值-真实值
                    metrics=['accuracy']
                    )
    # epochs训练5个批次,validation_freq每2个批次验证1次。
    history = network.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2)
    print(history.history)  # 历史训练准确率
    # 1.保存网络模型，保存的是张量的形式
    # network.save_weights('weight.ckpt')
    # del network  # 删除网络
    # 模型的加载
    # network.load_weights('weight.ckpt')
    # 2.保存网络的结构信息以及参数，不需要提前创建相同的网络对象。
    # network.save('model.h5')
    # keras.models.load_model('model.h5')
    # 3.跨平台保存 ,保存为pb格式
    # tf.saved_model.save(network, 'model-savedmodel')
    # tf.saved_model.load('model-savedmodel')

    # 准确率计量器
    # acc_meter = metrics.CategoricalAccuracy()
    # for x, y in val_db:
    #     # 前向计算
    #     pred = network(x)
    #     acc_meter.update_state(y_true=y, y_pred=y)
    # print(acc_meter.result())


