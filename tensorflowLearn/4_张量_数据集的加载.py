# coding: utf-8 
# @时间   : 2021/9/8 11:20 上午
# @作者   : 文山
# @邮箱   : wolaizhinidexin@163.com
# @作用   :
# @文件   : 4_张量_数据集的加载.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

if __name__ == "__main__":
    # 自动加载imdb电影评价数据集
    # 三维张量[b,sequence len,feature len], 信号的数量, 步数，特征长度
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=1000)
    # 自动截断成80个句子
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
    print(x_train.shape)
    # 将数字编码的单词转换成长度为100个词的向量
    embedding = layers.Embedding(10000, 100)
    out = embedding(x_train)
    print(out.shape)
