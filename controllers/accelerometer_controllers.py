from datetime import datetime, timedelta
from decimal import Decimal
import os
import time
from flask import Flask, render_template, url_for, request, redirect, session, jsonify, send_file, Blueprint, make_response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from model import db, Accelerometer
import jwt
from config import SECRET_KEY, UPLOAD_FOLDER
from configs.utils import *
from sqlalchemy import func
import sqlalchemy as sa

accelerometer_router = Blueprint('accelerometer_router', __name__)


@accelerometer_router.route('/api/accelerometer', methods=['GET', 'POST'])
@token_required
def accelerometer_handler(current_user):
    if request.method == 'POST':
        try:
            submitted_file = request.files['file']
            if submitted_file and allowed_filename(submitted_file.filename):
                filename = secure_filename(
                    submitted_file.filename)
                split_filename = '.' in filename and filename.rsplit('.', 1)
                filename = ''.join(
                    split_filename[:-1]) + "_" + str(time.time()) + "." + split_filename[-1]
                os.makedirs(os.path.dirname(UPLOAD_FOLDER), exist_ok=True)
                submitted_file.save(os.path.join(UPLOAD_FOLDER, filename))
            else:
                return jsonify({"error": "Exception: {}".format(e)}), 404

            with open(os.path.join(UPLOAD_FOLDER, filename)) as f:
                data = f.readlines()
                for line in data:
                    arr = line.split(" ")
                    x = Decimal(arr[0])
                    y = Decimal(arr[1])
                    z = Decimal(arr[2])
                    time_string = arr[3].strip().replace("T", " ")
                    
                    timestamp = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
                    acc = Accelerometer(
                        x=x, y=y, z=z, timestamp=timestamp, id_user=current_user.id)
                    db.session.add(acc)
                    db.session.commit()

            res = {
                'msg': "Successfully!"
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
