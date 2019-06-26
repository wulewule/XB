'''
为哲学家设立三种状态
只有饥饿状态可以拿起
叉子----线程设立多种
信号量控制多种状态来
避免死锁

'''
import threading 
import time

class death():

    x = [True, True, True, True, True]
    can = [0, 0, 0, 0, 0]
    mutex = {'1':'H', '2':'H', '3':'H', '4':'H', '5':'H'}
    
    def makelock(self, num):
        
        if self.x[num-1]:
            self.x[num-1] = False
            print(str(num) + '号哲学家拿起左边' + str(num-1) + '号叉子')
            self.can[num-1] += 1
        
        if num == 5:
            if self.x[0]:
                self.x[0] = False
                print(str(num) + '号哲学家拿起左边0号叉子')
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
            print(str(num) + '号哲学家放下左边0号叉子')
            self.can[num-1] -= 1
        
        else:
            self.x[num] = True
            print(str(num) + '号哲学家放下右边' + str(num) + '号叉子')
            self.can[num-1] -= 1
    
    def eat(self, num):
        
        while 1:
            if self.mutex[str(num)] == 'H':
                self.makelock(num)
                if self.can[num-1] == 2:
                    print(str(num)+'号哲学家进餐状态')
                    self.mutex[str(num)] = 'J'
                    time.sleep(1)
                    self.golock(num)
                    self.mutex[str(num)] = 'S'
                    print(str(num)+'号哲学家思考状态')
            else:
                time.sleep(2)
                self.mutex[str(num)] = 'H' 

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
