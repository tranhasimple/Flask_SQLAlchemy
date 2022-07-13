from flask import Flask, render_template, url_for, request, redirect, session, jsonify, send_file, Blueprint
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import User, db

auth_router = Blueprint('auth_router', __name__)


@auth_router.route('/api/register', methods=["POST"])
def register_func():
    try:
        username = request.form['username']
        password = request.form['password']
        password = generate_password_hash(password)
        gender = request.form['gender']
        birthday = request.form['birthday']
        user = User(username=username, password=password, gender=gender,
                    birthday=birthday)
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        return jsonify({"error": "Exception: {}".format(e)}), 400
    return jsonify(user.toDict()), 200


@auth_router.route('/api/login', methods=["POST"])
def login_func():
    try:
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if check_password_hash(user.password, password):
            return jsonify(user.toDict()), 200
        else:
            return jsonify({
                "msg": "Something went wrong"
            }), 400
    except Exception as e:
        return jsonify({'Error': "{}".format(e)}), 500
