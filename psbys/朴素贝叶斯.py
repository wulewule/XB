# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

# 广告、垃圾标识
adClass = 1


def loadDataSet():
    """加载数据集合及其对应的分类"""
    wordsList = [['周六', '公司', '一起', '聚餐', '时间'],
                 ['优惠', '返利', '打折', '优惠', '金融', '理财'],
                 ['喜欢', '机器学习', '一起', '研究', '欢迎', '贝叶斯', '算法', '公式'],
                 ['公司', '发票', '税点', '优惠', '增值税', '打折'],
                 ['北京', '今天', '雾霾', '不宜', '外出', '时间', '在家', '讨论', '学习'],
                 ['招聘', '兼职', '日薪', '保险', '返利']]
    # 1 是, 0 否
    classVec = [0, 1, 0, 1, 0, 1]
    return wordsList, classVec


# python中的& | 是位运算符   and or是逻辑运算符
# 当and 为 true时，返回的并不是true而是运算结果最后一个变量值
# 当and 为 false时，返回的是第一个false的值，如果a为false 则返回a，如果a不是false，那么返回b
# 当a or b 为true时，返回的是第一个真的变量值，如果a,b都为真时候那么返回a 如果a为假b为真那么返回b
# a & b a和b为两个set,返回结果取a和b的交集  a|b a和b为两个set,返回结果为两个集合的不重复并集--重点是不重复，最方便去重
def doc2VecList(docList):
    # 从第一个和第二个集合开始进行并集操作，最后返回一个不重复的并集
    # reduce--对可迭代对象逐一操作并累积
    a = list(reduce(lambda x, y: set(x) | set(y), docList))
    return a
    # 得到的a是所有数据的列表


def words2Vec(vecList, inputWords):
    """把敏感词统一转化为词向量"""
    # 转化成以一维数组
    resultVec = [0] * len(vecList)
    for word in inputWords:
        if word in vecList:
            # 在单词出现的位置上的计数加1
            resultVec[vecList.index(word)] += 1
        else:
            print('没有发现此单词')

    return np.array(resultVec)  # ---转为numpy格式


def trainNB(trainMatrix, trainClass):
    """计算，生成每个词对于类别上的概率"""
    # 类别行数
    numTrainClass = len(trainClass)
    # 列数
    numWords = len(trainMatrix[0])  # 计算第一行即可，因为各个列表长度相同

    p0Num = np.ones(numWords)  # 概率会出错，只能初始化为一，数据多的时候影响小
    p1Num = np.ones(numWords)
    # 相应的单词初始化为2
    p0Words = 2.0
    p1Words = 2.0
    # 统计每个分类的词的总数
    # 训练数据集的行数作为遍历的条件，从1开始
    # 如果当前类别为1，那么p1Num会加上当前单词矩阵行数据，依次遍历
    # 如果当前类别为0，那么p0Num会加上当前单词矩阵行数据，依次遍历
    # 同时统计当前类别下单词的个数和p1Words和p0Words
    for i in range(numTrainClass):
        if trainClass[i] == 1:  # 即列表为敏感词列表
            # 数组在对应的位置上相加
            p1Num += trainMatrix[i]  # 这个时候为矩阵相加
            p1Words += sum(trainMatrix[i]) # 敏感词出现次数
        else:
            p0Num += trainMatrix[i]  #同理，为非敏感词出现次数计算
            p0Words += sum(trainMatrix[i])
    # 计算每种类型里面， 每个单词出现的概率
    # 朴素贝叶斯分类中，y=x是单调递增函数，y=ln(x)也是单调的递增的
    # 如果x1>x2 那么ln(x1)>ln(x2)
    # 在计算过程中，由于概率的值较小，所以我们就取对数进行比较，根据对数的特性
    # ln(MN) = ln(M)+ln(N)
    # ln(M/N) = ln(M)-ln(N)
    # ln(M**n)= nln(M)
    p0Vec = np.log(p0Num / p0Words) # 均为矩阵计算
    p1Vec = np.log(p1Num / p1Words)
    # 计算在类别中1出现的概率，0出现的概率可通过1-p得到
    pClass1 = sum(trainClass) / float(numTrainClass)
    return p0Vec, p1Vec, pClass1


def classifyNB(testVec, p0Vec, p1Vec, pClass1):
    # 据说用ln会简化计算，用不用影响不大，就是概率大小比较，二选一
    p1 = sum(testVec * p1Vec) + np.log(pClass1)
    p0 = sum(testVec * p0Vec) + np.log(1 - pClass1)
    if p0 > p1:
        return 0
    return 1


def printClass(words, testClass):
    if testClass == adClass:
        print(words, '推测为：广告邮件')
    else:
        print(words, '推测为：正常邮件')


def tNB():
    # 从训练数据集中提取出属性矩阵和分类数据
    docList, classVec = loadDataSet()
    # 生成包含所有单词的list
    # 此处生成的单词向量是不重复的
    allWordsVec = doc2VecList(docList)
    # 构建词向量矩阵
    # 计算docList数据集中每一行每个单词出现的次数，其中返回的trainMat是一个数组的数组
    trainMat = list(map(lambda x: words2Vec(allWordsVec, x), docList))
    # 训练计算每个词在分类上的概率, p0V:每个单词在非分类出现的概率， p1V:每个单词在是分类出现的概率
    # 其中概率是以ln进行计算的
    # pClass1为类别中是1的概率
    p0V, p1V, pClass1 = trainNB(trainMat, classVec)
    # 测试数据集
    testWords = ['公司', '聚餐', '讨论', '贝叶斯']
    testVec = words2Vec(allWordsVec, testWords)
    # 判断当前数据的分类情况
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    # 打印出测试结果
    printClass(testWords, testClass)

    testWords = ['公司', '保险', '金融']
    testVec = words2Vec(allWordsVec, testWords)
    # 判断当前数据的分类情况
    testClass = classifyNB(testVec, p0V, p1V, pClass1)
    # 打印出测试结果
    printClass(testWords, testClass)


tNB()
