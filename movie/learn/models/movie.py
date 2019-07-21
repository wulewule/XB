import sys
sys.path.append('E:\\Spider\\movie')
from sqlalchemy import Column, String
from flask_sqlalchemy import SQLAlchemy 

db = SQLAlchemy()

#建立模型
class Movie(db.Model):

    name = Column(String(100), primary_key=True) #影片名字
    link = Column(String(100)) #影片磁力链接
    image = Column(String(200)) #影片封面
    introduction = Column(String(500)) #影片介绍
    #default-初始
    #unique-唯一性                                                               