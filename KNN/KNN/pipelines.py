# -*- coding: utf-8 -*-

from scrapy.exceptions import DropItem
import json
from Emotion.KNN import KNN

class MyPipeline(object):
    #该方法用于处理数据
    def process_item(self, item, spider):
        #读取item中的数据
        KNN.Question.append(item['data'])
        KNN.Result.append(item['analyse'])
        #返回item
        return item
    #该方法在spider被开启时被调用。
    def open_spider(self, spider):
        print('开始爬取题目信息--------')
    
    #该方法在spider被关闭时被调用。
    def close_spider(self, spider):
        a = KNN()
        a.get()