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

    def __repr__(self) -> str:
        return f"<date> {self.timestamp}" 


class ResponseData(db.Model):
    __tablename__ = "response_datas"

    id = db.Column(db.Integer, primary_key=True)
    steps = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime)

    def toDict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }
