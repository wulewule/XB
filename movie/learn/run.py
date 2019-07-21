import sys
sys.path.append('E:\\Spider\\movie')
from learn import creat_app
from flask_sqlalchemy import SQLAlchemy

app = creat_app()

app.run()