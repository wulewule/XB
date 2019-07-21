import sys
sys.path.append('E:\\Spider\\movie')
from flask import Flask
from learn.models.movie import db

def creat_app():
    app = Flask(__name__)
    
    app.config.from_object('learn.secure')
    app.config.from_object('learn.setting')
    
    register_blueprint(app)
    db.init_app(app)
    db.create_all(app=app)
    return app

def register_blueprint(app):
    from learn.web.movie import web
    app.register_blueprint(web)