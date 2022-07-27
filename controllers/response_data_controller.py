from datetime import datetime, date
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
                steps=steps, timestamp=timestamp, user_id=id_user)
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
            query_date = request.args.get('date')
            if query_date == None:
                return jsonify({
                    "msg": "No date found"
                }), 404

            query_date = datetime. strptime(query_date, '%Y-%m-%d')
            query_date = date(int(query_date.year), int(
                query_date.month), int(query_date.day))

            data = ResponseData.query.all()

            res = 0
            for acc in data:
                ts = acc.timestamp
                if query_date.year == ts.year and query_date.month == ts.month and query_date.day == ts.day:
                    res += acc.steps
                    
                    # res.append({
                    #     "steps": acc.steps,
                    #     "hour": ts.hour,
                    #     "timestamp": ts
                    # })
        except Exception as e:
            return jsonify({"error": "Exception: {}".format(e)}), 400
        return jsonify(res), 200
