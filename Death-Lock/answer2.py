'''
奇数位的哲学家先拿左手叉子
后拿右手叉子
偶数位的哲学家先拿右手叉子
后拿左手叉子
为进程进行编号，按奇偶数进
行就餐

'''
import threading 
import time

class death():

    x = [True, True, True, True, True]
    can = [0, 0, 0, 0, 0]
    
    def makelockl(self, num):

        if self.x[num-1]:
            self.x[num-1] = False
            print(str(num) + '号哲学家拿起左边' + str(num-1) + '号叉子')
            self.can[num-1] += 1

    def makelockr(self, num):

        if num == 5:  
           if self.x[0]:
                self.x[0] = False
                print(str(num) + '号哲学家拿起右边0号叉子')

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
            
            time.sleep(1)
            
            if sum%2 == 0:
                self.makelockr(sum)
                self.makelockl(sum)
            else:
                self.makelockl(sum)
                self.makelockr(sum)
            
            if self.can[sum-1] == 2:
                print(str(sum)+'号哲学家进餐--')
                time.sleep(2)
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
