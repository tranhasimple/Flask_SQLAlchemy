from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

import os

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:Long.0311@localhost:3306/acc_data"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

Session = sessionmaker()
Base = declarative_base()