import math
import operator

def creatDataSet():
    
    labels = ['no surfacing', 'flippers']
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    return dataSet,labels

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
    return classCount[0][0]  #返回最大值                    #reverse--True--倒序

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
    print(bestFeatLable)
    myTree = {bestFeatLable:{}} #myTree 初始化
    del(labels_copy[bestFeat])              #用某个特征类别划分数据集后，把这个特征类别从labels中删除
    featValues = [example[bestFeat] for example in dataSet] #返回dataSet中bestFeat特征的那一列数据
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels_copy[:]
        newDataSet = splitDataSet(dataSet, bestFeat, value) #用[信息增益]最高的特征类别来划分数据集
        myTree[bestFeatLable][value] = createTree(newDataSet,subLables) #递归创建决策树
        print(myTree[bestFeatLable][value])
        #print('www')
    return myTree

def classify(inputTree,featLables,testVec): # 三个参数：决策树，特征类别列表，测试数据
    
    firstStr = list(inputTree.keys())[0]    # 获取树的第一个判断块(即根节点)
    print(firstStr)
    secondDict = inputTree[firstStr]        # 第一个判断块的子树
    featIndex = featLables.index(firstStr)  # index查找firstStr
    for key in secondDict.keys():           # key是数据里的是或否
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__== 'dict':                     #若还有dict字典变量（即若还有子树）
                classLable = classify(secondDict[key],featLables,testVec)   #递归，从上往下遍历决策树
            else:
                classLable = secondDict[key]
    return classLable


dataSet, labels = creatDataSet()    #获取数据集
tree = createTree(dataSet, labels)  #形成决策树
print(tree)                         #打印决策树
ret = classify(tree,labels,[1,1])   #数据测试
#print(ret)     