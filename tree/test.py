'''
不同于树，我们不用考虑是否每个分支的结果
只需考虑对数据的处理，让程序自己建树
'''
'''
熵：表示随机变量的不确定性--越小说明越准确
条件熵：在一个条件下，随机变量的不确定性
信息增益：熵 - 条件熵
在一个条件下，信息不确定性减少的程度
例如明天是下雨天信息熵是2， 条件熵是0.01（阴天）
就是如果是因阴天，明天下雨的不确定性减少了1.99
信息增益则为1.99 所以阴天这个条件很重要
'''

import csv
import math
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#定义自定义字体，文件名是系统中文字体
myfont = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simkai.ttf') 
#解决负号'-'显示为方块的问题 
matplotlib.rcParams['axes.unicode_minus']=False

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def creatDataSet():
    
    dataSet = []
    with open('adult.csv', encoding = 'utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",")[0:100]
    data_row = [0, 3, 5, 14]
    data = data[:, data_row]
    for x in data:
        x = list(x)
        if int(x[0])>10 and int(x[0])<=20:
            x[0] = "10~20"
        elif int(x[0])>20 and int(x[0])<=30:
            x[0] = "20~30"
        elif int(x[0])>30 and int(x[0])<=40:
            x[0] = "30~40"
        elif int(x[0])>40 and int(x[0])<=50:
            x[0] = "40~50"
        else:
            x[0] = "50+"
        dataSet.append(x)
    #print(dataSet)
    labels = ['age', 'education', 'marital-status']	
    return dataSet, labels

def calcShannonEnt(dataSet):

    numEntries = len(dataSet)               #计算出数据集有多少组数据
    labelCount ={}                          #用于记录当前数据集最后一列各个取值的出现次数
    for featVec in dataSet:                 #依次取出数据集的每组数据
        CurrentLabel = featVec[-1]          #获取每组数据最后一列数据(即数据集最后一列的数据)
        if CurrentLabel not in labelCount.keys():   #记录最后一列数据各个取值的出现次数
            labelCount[CurrentLabel]=0              #若某取值第一次出现，则加入到labelCount中，并设置次数为0
        labelCount[CurrentLabel]+=1                 #若某取值已经出现过，则次数加1
    #计算香农熵
    shannonEnt = 0.0                                #香农熵初始设置为0，0.0用于小数计算
    for key in labelCount:                          #取出数据集最后一列的各个取值
        prob = float(labelCount[key])/numEntries    #某个取值在数据集中的出现概率
        shannonEnt -= prob*math.log(prob,2)              #通过公式计算香农熵
        #print(shannonEnt)
    return shannonEnt

def splitDataSet(dataSet,axis,value):   # 3个参数分别是[要划分的数据集]，[特定特征类别]，[特定特征类别的某个取值]
    
    retDataSet=[]                           #要返回的子数据集，同理类似于操作形参
    for featVec in dataSet:                 #依此取出要划分数据集的每组数据
        if featVec[axis]==value:            #若某组数据的[特定特征类别]的取值与 value相等
            reduceFeatVec = featVec[:axis]              #则该组数据舍弃[特定特征类别] 那一列数据
            reduceFeatVec.extend(featVec[axis+1:])      #extend 追加到末尾，列表合并
            retDataSet.append(reduceFeatVec)            #获得新的数据集
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    
    numFeature = len(dataSet[0])-1          #得出数据集的[特征类别个数]，-1是因为最后一列是分类，所以不要
    baseEntropy = calcShannonEnt(dataSet)   #计算当前数据集的信息熵
    bestInfoGain = 0                        #初始信息增益 设为0
    bestFeature = -1                        #信息增益最大的 [特征类别] 初始设为 -1
    for i in range(numFeature):             #遍历各个[特征类别]
        featList=[number[i] for number in dataSet]      #取出序号为i列 的[特征类别]的值
        uniqueVals = set(featList)                      #集合set中的值不能相同，set是list去重的最有效方法
        newEntropy = 0
        for value in uniqueVals:                         #依次用序号为i的[特征类别]的值(value)来划分数据集
            subDataSet = splitDataSet(dataSet,i,value)   #返回{用序号为i的[特征类别] 的取值=value}这条件来划分的子数据集
            prob = len(subDataSet)/float(len(dataSet))   #{序号为i的[特征类别] 的取值=value}的子数据集的数据占父数据集的比例
            newEntropy +=prob*calcShannonEnt(subDataSet) #各个子数据集的[香农熵*子集合在父集合中的出现几率]之和
        InfoGain = baseEntropy - newEntropy  #序号为i的特征类别的[信息增益]
                                             #为父数据集的香农熵减去以序号为i的特征类别划分的各个子数据集的香农熵之和*出现几率
        #最大信息增益
        if(InfoGain > bestInfoGain):    #选出拥有最大[信息增益]的特征类别
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature 

def majorityCnt(classList):
    #此函数是找频率最高的类别并返回 sorted----对字典进行排序
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0  #如果不存在会自动创建
        classCount[vote] += 1
    sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #key=operator.itemgetter(1)----为一个确定比哪项的函数
    print(classCount)
    return list(classCount)[0] #返回最大值                    #reverse--True--倒序

def createTree(dataSet,labels):                # 接收数据集，标签
    
    labels_copy = labels[:]  #避免对原数据进行更改，类似于只操作形参                
    classList=[example[-1] for example in dataSet]  #把数据集的最后一个，即分类放在列表中
    #第一种情况，剩余类别完全相同
    if classList.count(classList[-1]) == len(classList):
        return classList[-1]
    #第二种情况，都已经删完了剩一个，返回频率最高的类别-----classList[-1]即分类
    if len(dataSet[0]) == 1:  #因为dataSet里面各个列表长度相同，所以判断dataSet[0]即可
        return majorityCnt(classList)
    #按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet) #返回[信息增益]最高的特征类别序号
    bestFeatLable = labels_copy[bestFeat] #该特征类别
    #print(bestFeatLable)
    myTree = {bestFeatLable:{}} #myTree 初始化
    del(labels_copy[bestFeat])              #用某个特征类别划分数据集后，把这个特征类别从labels中删除
    featValues = [example[bestFeat] for example in dataSet] #返回dataSet中bestFeat特征的那一列数据
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels_copy[:]
        newDataSet = splitDataSet(dataSet, bestFeat, value) #用[信息增益]最高的特征类别来划分数据集
        myTree[bestFeatLable][value] = createTree(newDataSet,subLables) #递归创建决策树
        #print(myTree[bestFeatLable][value])
    return myTree

def classify(inputTree,featLables,testVec): # 三个参数：决策树，特征类别列表，测试数据
    
    firstStr = list(inputTree.keys())[0]    # 获取树的第一个判断块(即根节点)
    secondDict = inputTree[firstStr]        # 第一个判断块的子树
    featIndex = featLables.index(firstStr)  # index查找firstStr
    for key in secondDict.keys():           # key是数据里的是或否
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__== 'dict':                     #若还有dict字典变量（即若还有子树）
                classLable = classify(secondDict[key],featLables,testVec)   #递归，从上往下遍历决策树
            else:
                classLable = secondDict[key]
    return classLable

def run():
    
    dataSet, labels = creatDataSet()    #获取数据集
    tree = createTree(dataSet, labels)  #形成决策树
    print(tree)                         #打印决策树
    create_plot(tree)
    # ret = classify(tree,labels,[1,1])   #数据测试
    # print(ret)

def retrieve_tree(i):
    
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                    ]
    return list_of_trees[i]
 
def get_num_leafs(mytree):
    '''
    获取叶子节点数
    '''
    num_leafs = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
            
    return num_leafs
 
def get_tree_depth(mytree):
    '''
    获取树的深度
    '''
    max_depth = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    
    for key in second_dict.keys():
        # 如果子节点是字典类型，则该节点也是一个判断节点，需要递归调用
        # get_tree_depth()函数
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
            
        if this_depth > max_depth:
            max_depth = this_depth
            
    return max_depth
 
def plot_node(ax, node_txt, center_ptr, parent_ptr, node_type):
    '''
        绘制带箭头的注解
    '''
    ax.annotate(node_txt, xy=parent_ptr, xycoords='axes fraction',
                xytext=center_ptr, textcoords='axes fraction',
                va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

def plot_mid_text(ax, center_ptr, parent_ptr, txt):
    '''
    在父子节点间填充文本信息
    '''
    x_mid = (parent_ptr[0] - center_ptr[0]) / 2.0 + center_ptr[0]
    y_mid = (parent_ptr[1] - center_ptr[1]) / 2.0 + center_ptr[1]
 
    ax.text(x_mid, y_mid, txt)
 
def plot_tree(ax, mytree, parent_ptr, node_txt):
    '''
    绘制决策树
    '''
    # 计算宽度
    num_leafs = get_num_leafs(mytree)
    
    first_str = list(mytree.keys())[0]
    center_ptr = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_width, plot_tree.y_off)
    
    #绘制特征值，并计算父节点和子节点的中心位置，添加标签信息
    plot_mid_text(ax, center_ptr, parent_ptr, node_txt)
    plot_node(ax, first_str, center_ptr, parent_ptr, decision_node)
    
    second_dict = mytree[first_str]
    #采用的自顶向下的绘图，需要依次递减Y坐标
    plot_tree.y_off -= 1.0 / plot_tree.total_depth
    
    #遍历子节点，如果是叶子节点，则绘制叶子节点，否则，递归调用plot_tree()
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            plot_tree(ax, second_dict[key], center_ptr, str(key))
        else:
            plot_tree.x_off += 1.0 / plot_tree.total_width
            plot_mid_text(ax, (plot_tree.x_off, plot_tree.y_off), center_ptr, str(key))
            plot_node(ax, second_dict[key], (plot_tree.x_off, plot_tree.y_off), center_ptr, leaf_node)
    
    #在绘制完所有子节点之后，需要增加Y的偏移
    plot_tree.y_off += 1.0 / plot_tree.total_depth
 
def create_plot(in_tree):
    
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    
    ax_props = dict(xticks=[], yticks=[])
    ax = plt.subplot(111, frameon=False, **ax_props)
    plot_tree.total_width = float(get_num_leafs(in_tree))
    plot_tree.total_depth = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_width
    plot_tree.y_off = 1.0
    plot_tree(ax, in_tree, (0.5, 1.0), "")
    plt.show()

# myTree=retrieve_tree(0)
# create_plot(myTree)
# print(myTree)
run()