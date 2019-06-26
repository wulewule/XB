'''
只允许四位哲学家同时进餐
一位哲学家用餐完毕另一位
填补他的位置----类比只允s
许不产生死锁的几个线程操
作资源，其他的进行等待

'''
import threading 
import time

class death():

    x = [True, True, True, True, True]
    can = [0, 0, 0, 0, 0]
    sum = 0
    
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
    
    def eat(self, num):
        
        while 1:
            if self.sum < 4:
                self.sum += 1
                while 1:    
                    time.sleep(1)
                    self.makelock(num)
                    
                    if self.can[num-1] == 2:
                        print(str(num)+'号哲学家进餐--')
                        time.sleep(2)
                        self.golock(num)
                        self.sum -= 1
                        break
                    
                    else:
                        print(str(num)+'号哲学家等待进餐--')  

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
