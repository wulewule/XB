import scrapy
#引入容器
from Emotion.items import EmotionItem


class MySpider(scrapy.Spider):
    #设置name
    name = "MySpider"
    #设定域名
    allowed_domains = ["huazhen2008.com"]
    #填写爬取地址
    start_urls = ["https://www.huazhen2008.com/article/42830.html"]
    #编写爬取方法
    
    def parse(self, response):
        
        number = 0
        item = EmotionItem()
        #实例一个容器保存爬取的信息
        html_data = response.xpath('//div[@class="page_cvans"]/p//text()').extract()
 
        for data in html_data:
            number += 1
 
            if(number >=4 and number <=23):
                item['data'] = data.strip() 
                yield item
            
            if(number >28 and number <=38):
                item['data']=''
                item['analyse'] = data.strip()
                yield item
           