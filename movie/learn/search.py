import requests
import re, time

class data_search():

    names = [] #电影名字
    images = [] #电影封面
    links= [] #磁力链接
    introductions = [] #电影介绍
    
    @classmethod
    def data_get(cls):

        #获取每一页的链接与电影名字
        for page in range(1,190):
            url='http://www.ygdy8.net/html/gndy/dyzz/list_23_'+str(page)+'.html'
            time.sleep(5)
            all_html = requests.get(url)
            all_html.encoding='gb2312'#指定编码
            page_link = re.findall('<a href="(.*?)" class="ulink">', all_html.text)
            movie_name = re.findall('<a href="[\s\S]*?" class="ulink">([\s\S]*?)</a>', all_html.text)
            
            #访问每一页
            for m in page_link:
                time.sleep(1)
                page_url = 'http://www.ygdy8.net'+m
    
                time.sleep(5)
                page_html = requests.get(page_url)
                page_html.encoding='gb2312'#指定编码

                #获取影片封面       
                image = re.findall('<img border="0" src="(.*?)" alt="" />', page_html.text)
                if not image:
                    image = re.findall('<img border="0" alt="" src="(.*?)" />', page_html.text)
                if image:
                    cls.images.append(image[0])
                else:
                    cls.images.append('暂无封面显示')
            
                #获取影片磁力链接
                link = re.findall('<a href="(.*?)">ftp://.*?</a></td>', page_html.text)
                if link:
                    cls.links.append(link)
                else:
                    cls.links.append('暂无播放链接')
                
                #获取影片介绍
                introduction = list(filter(None, re.findall('<br />([\s\S]*?)<br />', re.sub('\u3000', '', page_html.text))))[1:-3]
                introduction = ''.join(introduction)
                if introduction:
                    cls.introductions.append(introduction)
                else:
                    cls.introductions.append('暂无介绍')
            
            for x in movie_name:
                print(x)
                cls.names.append(x)
        return cls.names, cls.links, cls.images, cls.introductions
# a = data_search()
# a.data_get()