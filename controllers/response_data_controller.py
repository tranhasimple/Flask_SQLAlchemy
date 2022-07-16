from datetime import datetime, timedelta
from decimal import Decimal
import os
import time
from flask import Flask, render_template, url_for, request, redirect, session, jsonify, send_file, Blueprint, make_response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import ResponseData, db, Accelerometer
import jwt
from config import SECRET_KEY, UPLOAD_FOLDER
from configs.utils import *
from sqlalchemy import func
import sqlalchemy as sa
import csv

response_router = Blueprint('response_router', __name__)


@response_router.route('/api/response', methods=['GET', 'POST'])
@token_required
def response_handler(current_user):
    if request.method == 'POST':
        try:
            steps = request.form["steps"]
            timestamp = request.form["timestamp"]
            id_user = current_user.id

            response_data = ResponseData(
                steps=steps, timestamp=timestamp, id_user=id_user)
            db.session.add(response_data)
            db.session.commit()
            res = {
                'msg': response_data.toDict()
            }
        except Exception as e:
            return jsonify({"error": "Exception: {}".format(e)}), 400
        return jsonify(res), 200
    elif request.method == 'GET':
        try:
            data = Accelerometer.query.filter(
                Accelerometer.id_user == current_user.id).group_by(func.date_format(Accelerometer.timestamp, '%Y-%m-%d %H')).all()

            res = []
            for acc in data:
                res.append(acc.toDict())
        except Exception as e:
            return jsonify({"error": "Exception: {}".format(e)}), 400
        return jsonify(res), 200
