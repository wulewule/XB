# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class EmotionItem(scrapy.Item):
    
    #情感题目
    data = scrapy.Field()
    #情感分析
    analyse = scrapy.Field()