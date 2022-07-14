from __future__ import absolute_import
from flask_sqlalchemy import SQLAlchemy
from dataclasses import dataclass
from sqlalchemy import inspect

db = SQLAlchemy()


class Accelerometer(db.Model):
    __tablename__ = "accelerometers"

    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    z = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    user_id = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f"<date> {self.timestamp}" 


class ResponseData(db.Model):
    __tablename__ = "response_datas"

    id = db.Column(db.Integer, primary_key=True)
    steps = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime)
    user_id = db.Column(db.Integer)

    def toDict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }

class User(db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200))
    gender = db.Column(db.Integer)
    birthday = db.Column(db.DateTime)
    
    def toDict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }
