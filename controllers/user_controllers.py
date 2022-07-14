from datetime import datetime, timedelta
from flask import Flask, render_template, url_for, request, redirect, session, jsonify, send_file, Blueprint, make_response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import User, db
import jwt
from config import SECRET_KEY
from configs.utils import *
auth_router = Blueprint('auth_router', __name__)


@auth_router.route('/api/register', methods=["POST"])
def register_func():
    try:
        username = request.form['username']
        password = request.form['password']
        password = generate_password_hash(password)
        gender = request.form['gender']
        birthday = request.form['birthday']

        user = User.query\
            .filter_by(username=username)\
            .first()
        if not user:

            user = User(username=username, password=password, gender=gender,
                        birthday=birthday)

            db.session.add(user)
            db.session.commit()
            return jsonify(user.toDict()), 200
        else:
            return make_response('User already exists. Please Log in.', 202)

    except Exception as e:
        return jsonify({"error": "Exception: {}".format(e)}), 500


# get username and password from user -> call to db and filter and check username
# - if username is valid --> check code_hash --> duration 10000 minutes
@auth_router.route('/api/login', methods=["POST"])
def login_func():
    try:
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if not user:
            return make_response(
                'Could not verify',
                401
            )

        if check_password_hash(user.password, password):
            token = jwt.encode({
                'username': user.username,
                'exp': datetime.utcnow() + timedelta(minutes=10000)
            }, SECRET_KEY)
            return make_response(jsonify({'token': token.decode('UTF-8'), 'user': user.toDict()}), 201)
        else:
            return jsonify({
                "msg": "Something went wrong"
            }), 400
    except Exception as e:
        return jsonify({'Error': "{}".format(e)}), 500


@auth_router.route('/api/me', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({'users': current_user.toDict()})
