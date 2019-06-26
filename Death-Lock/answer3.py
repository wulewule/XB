'''
破坏规则，只有能就餐的
哲学家才能拿叉子，即不
能只拿一只叉子，只拿一
只的放下--出现死锁释放
自己的资源，等待

'''
import threading 
import time

class death():

    x = [0, 0, 0, 0, 0]
    can = [0, 0, 0, 0, 0]
    
    def makelock(self, num):
        
        if not self.x[num-1]:
            self.x[num-1] = num
            print(str(num) + '号哲学家拿起左边' + str(num-1) + '号叉子')
            self.can[num-1] += 1
        
        if num == 5:
            if not self.x[0]:
                self.x[0] = num
                print(str(num) + '号哲学家拿起右边0号叉子')
                self.can[num-1] += 1
        
        else:
            if not self.x[num]:
                self.x[num] = num
                print(str(num) + '号哲学家拿起右边' + str(num) + '号叉子')
                self.can[num-1] += 1
    
    def golock(self, num):

        if self.x[num-1] == num: 
            self.x[num-1] = 0
            print(str(num) + '号哲学家放下左边' + str(num-1) + '号叉子')
            self.can[num-1] -= 1

        if num == 5:
            if self.x[0] == num:
                self.x[0] = 0
                print(str(num) + '号哲学家放下右边0号叉子')
                self.can[num-1] -= 1
        
        else:
            if self.x[num] == num:
                self.x[num] = 0
                print(str(num) + '号哲学家放下右边' + str(num) + '号叉子')
                self.can[num-1] -= 1
    
    def eat(self, num):

        while 1:
            time.sleep(1)
            self.makelock(num)
            if self.can[num-1] == 2:
                print(str(num)+'号哲学家进餐--')
                time.sleep(2)
                self.golock(num)

            else:
                self.golock(num)
                print(str(num)+'1号哲学家等待进餐--')  
    
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
