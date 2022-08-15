from distutils.log import debug
from flask import Flask, request, jsonify, make_response
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from flask_restful import Api
from model import ResponseData, db
import config
from model import Accelerometer

from controllers.user_controllers import auth_router
from controllers.accelerometer_controllers import accelerometer_router
from controllers.response_data_controller import response_router
from controllers.acc_load_controller import acc_load

app = Flask(__name__)


app.config.from_object('config')
db.init_app(app=app)
api = Api(app)

app.register_blueprint(auth_router)
app.register_blueprint(accelerometer_router)
app.register_blueprint(response_router)
app.register_blueprint(acc_load)

if __name__ == "__main__":
    app.run(host=config.APP_HOST, port=config.APP_PORT,
            debug=True, threaded=True)
