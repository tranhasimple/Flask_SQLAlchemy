from re import T
from flask_cors import CORS
import logging
from Predict import StepPredict, Signal_new, predict

# from asyncio import FastChildWatcher
from datetime import datetime, timedelta
from decimal import Decimal
import os
import numpy as np

import time
import pandas as pd
import glob
from flask_cors import CORS
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

acc_load = Blueprint('acc_load', __name__)
HOME = "upload/by_hour/"

write_file = False


@acc_load.route('/api/acc_load', methods=['GET', 'POST'])
@token_required
def accelerometer_handler(current_user):
    global write_file
    if request.method == 'POST':
        if request.is_json:
            datas = request.json['data']
            sign = convert_data(datas)
            write_permission = request.json['writting']
            write_file = write_permission
            now = datetime.now()
# HERE=============================================

            if write_file:
                data_ = []
                hour = ""

                for item in datas:
                    data_.append(
                        [item['timestamp'], item['x'], item['y'], item['z']])

                file_path = os.path.join(HOME, str(current_user.id))
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                if(len(data_) > 0):
                    filename = data_[0]
                    hour = filename[0].split(
                        ":")[0] + "_" + filename[0].split(":")[1]

                    formated_file = hour.replace(" ", "-")

                    dir = os.path.join(file_path, formated_file + ".csv")

                    file_object = open(dir, 'a')
                    for item in data_:
                        text = item[0] + ";" + str(item[1]) + ";" + \
                            str(item[2]) + ";" + str(item[3]) + "\n"
                        file_object.write(text)
                    file_object.close()


    return jsonify({
        "message":"success"
    }), 200

def convert_data(data: list) -> list:
    lst = []
    # for i, e in enumerate(data):
    #     lst.append([data[i]['x'], data[i]['y'], data[i]['z']])
    for item in data:
        lst.append([item['x'], item['y'], item['z']])
    lst = np.array(lst)
    return lst