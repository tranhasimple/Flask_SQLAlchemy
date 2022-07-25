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

app = Flask(__name__)

# config = Config()

app.config.from_object('config')
db.init_app(app=app)
api = Api(app)

app.register_blueprint(auth_router)
app.register_blueprint(accelerometer_router)
app.register_blueprint(response_router)
# @app.route('/', methods=['GET'])
# def get():
#     try:
#         data = ResponseData.query.all()
#         res = []
#         for step in data:
#             res.append(step.toDict())

#     except Exception as e:
#         return jsonify({"error": "Exception: {}".format(e)}), 400

#     return jsonify(res)


if __name__ == "__main__":
    app.run(host=config.APP_HOST, port=config.APP_PORT,
            debug=True, threaded=True)
