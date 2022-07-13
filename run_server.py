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

app = Flask(__name__)

# config = Config()

app.config.from_object('config')
db.init_app(app=app)
api = Api(app)

app.register_blueprint(auth_router)

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


# @app.route('/', methods=['POST'])
# def post():
#     try:
#         x = request.form['x']
#         y = request.form['y']
#         z = request.form['z']
#         timestamp = request.form['timestamp']

#         acc = Accelerometer(x=x, y=y, z=z, timestamp=timestamp)

#         db.session.add(acc)
#         db.session.commit()
#         res = {
#             'x': acc.x,
#             'y': acc.y,
#             'z': acc.z,
#             'timestamp': acc.timestamp
#         }
#     except Exception as e:
#         return jsonify({"error": "Exception: {}".format(e)}), 400
#     return jsonify(res), 200


if __name__ == "__main__":
    app.run(host=config.APP_HOST, port=config.APP_PORT,
            debug=True, threaded=True)
