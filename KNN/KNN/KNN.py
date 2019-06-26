import random
import numpy as np
import matplotlib.pyplot as plt

class KNN():

    Question = []
    Result = []
    Data = []
    Data_result = []
    Ksh_classic = []
    Ksh_train = []
    Choose = []
    Choose_result = []
    
    def get(self):
        
        self.createdata()
        self.result_format()
        self.make_ksh()
        self.ksh()

        for x in self.Question:
            if x:
                print(x[0:-6]+'------*题为重点')
                choose = input('请输入你的程度指数 1~~5:\n')     
                self.Choose.append(int(choose))
        self.Choose = np.array(self.Choose) 
        self.analyse()
    
    def analyse(self):
        
        dataSize = self.Data.shape[0]

        diff = np.tile(self.Choose,(dataSize,1)) - self.Data
        sqdiff = diff ** 2
        squareDist = sum(sqdiff,axis = 1)
        dist = squareDist ** 0.5
    
        sortedDistIndex = np.argsort(dist)

        classCount={}
        for i in range(3):
            vote = self.Data_result[sortedDistIndex[i]]
            classCount[vote] = classCount.get(vote,0) + 1
        maxCount = 0
        for key,value in classCount.items():
            if value > maxCount:
                maxCount = value
                classes = key
        print('测试数据为'+str(self.Choose))
        print(self.Result[int(classes)])

    def createdata(self):
        
        data = []
        for i in range(30):     
            for j in range(20):
                x = random.randint(1,6)
                data.append(x)
            self.Data.append(data)
            self.result_add(data)
            data = []
        self.Data = np.array(self.Data)   

    def result_format(self):

        self.Result[0] = self.Result[1]+self.Result[2]+self.Result[3]
        self.Result[1] = self.Result[5]
        self.Result[2] = self.Result[7]
        self.Result[3] = self.Result[9]
        self.Result[4:] = []

    def result_add(self, data):
        
        if sum(data) >= 20 and sum(data) <= 40:
            self.Data_result.append(random.randint(0,5))
            self.Ksh_classic.append(1)

        elif sum(data) > 40 and sum(data) <= 60:
            self.Data_result.append(random.randint(5,10))
            self.Ksh_classic.append(2)

        elif sum(data) > 60 and sum(data) <= 80:
            self.Data_result.append(random.randint(10,15))
            self.Ksh_classic.append(3)

        else:
            self.Data_result.append(random.randint(15,20))
            self.Ksh_classic.append(4)
    
    def make_ksh(self):

        for i in range(30):
            ksh_train = []
            ksh_train.append(sum(self.Data[i]))
            ksh_train.append(self.Data_result[i])
            ksh_train.append(self.Ksh_classic[i])
            self.Ksh_train.append(ksh_train)
        print(self.Ksh_train)

    def ksh(self):
        
        point_dict={
            1 : '.',
            2 : ',',
            3 : 'o',
            4 : 'v',
            5 : '^',
        }
        
        color_dict={
            1 : 'b',
            2 : 'c',
            3 : 'g',
            4 : 'k',
            5 : 'm',
        }

        plt.figure(1,dpi=80)
        ax1=plt.subplot(221)
        ax2=plt.subplot(222)
        ax3=plt.subplot(223)

        for i in self.Ksh_train :
            plt.sca(ax1)
            plt.scatter(i[0],i[1], c=color_dict[i[2]], marker=point_dict[i[2]])
        for i in self.Ksh_train :
            plt.sca(ax3)
            plt.scatter(i[0],i[1], c=color_dict[i[2]], marker=point_dict[i[2]])

        for i in self.Ksh_train :
            plt.sca(ax2)
            plt.scatter(i[0],i[1], c=color_dict[i[2]], marker=point_dict[i[2]])
        for i in self.Ksh_train :
            plt.sca(ax3)
            plt.scatter(i[0],i[1], c=color_dict[i[2]], marker=point_dict[i[2]])
        plt.show()