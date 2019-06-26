'''
死锁模拟
哲学家拿起左右两只手的叉子
开始就餐，否则等待就餐，手
里的叉子只有就餐结束放下

'''
import threading 
import time

class death():

    x = [True, True, True, True, True]
    can = [0, 0, 0, 0, 0]

    def makelock(self, num):
        
        if self.x[num-1]:
            self.x[num-1] = False
            print(str(num) + '号哲学家拿起左边' + str(num-1) + '号叉子')
            self.can[num-1] += 1
        
        if num == 5:
            if self.x[0]:
                self.x[0] = False
                print(str(num) + '号哲学家拿起右边0号叉子')
                self.can[num-1] += 1
        
        else:
            if self.x[num]:
                self.x[num] = False
                print(str(num) + '号哲学家拿起右边' + str(num) + '号叉子')
                self.can[num-1] += 1
    
    def golock(self, num):

        self.x[num-1] = True
        print(str(num) + '号哲学家放下左边' + str(num-1) + '号叉子')
        self.can[num-1] -= 1
        
        if num == 5:
            self.x[0] = True
            print(str(num) + '号哲学家放下右边0号叉子')
            self.can[num-1] -= 1
        
        else:
            self.x[num] = True
            print(str(num) + '号哲学家放下右边' + str(num) + '号叉子')
            self.can[num-1] -= 1
    
    def eat(self, sum):
        
        while 1:
            time.sleep(0.1)
            self.makelock(sum)
            if self.can[sum-1] == 2:
                print(str(sum)+'号哲学家进餐--')
                time.sleep(0.2)
                self.golock(sum)

            else:
                print(str(sum)+'号哲学家等待进餐--')   
    
    def run(self):
       
        threads = []

        for i in range(1,6):
            t = threading.Thread(target=self.eat, args=(i,))
            threads.append(t)
        
        for x in threads:
            x.start()
        
        for y in threads:
            y.join()   

a = death()
a.run()
