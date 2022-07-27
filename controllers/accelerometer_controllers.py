from asyncio import FastChildWatcher
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

accelerometer_router = Blueprint('accelerometer_router', __name__)
HOME = "upload/by_hour/"

iter = 0
global_data = []
@accelerometer_router.route('/api/accelerometer', methods=['GET', 'POST'])
@token_required 
def accelerometer_handler(current_user):
    global iter
    global global_data
    if request.method == 'POST':
        if request.is_json:

            
            iter += 1
            datas = request.json['data']

            # print(datas)
            data_ = []
            hour = ""
            current_hour = ""

            for item in datas:
                data_.append([item['timestamp'],item['x'], item['y'], item['z']])

            # list all files in upload/by_hour
            # get lastest file => extract file_name
            # compare
            #glob

            # path of the directory
            path = "upload/by_hour"
            
            # Getting the list of directories
            dir = os.listdir(path)
            
            # Checking if the list is empty or not
            if len(dir) != 0:
        
                list_of_files = glob.glob('upload/by_hour/*') # * means all if need specific format then *.csv
                latest_file = max(list_of_files, key=os.path.getctime)
                print(type(latest_file))

                now = datetime.now()
                current_time = now.strftime("%y-%M-%d-%H_%M")

                print("current time: ", current_time)
                latest_hour = latest_file.split(".csv")[0].split(HOME)[1] 

                if latest_hour != current_time:
                    timestamp, x, y, z = np.genfromtxt(str(latest_file), delimiter=";", dtype='str',unpack=True)
                    x_d = x[1:]
                    y_d = y[1:]
                    z_d = z[1:]
                    x = np.array([float(item) for item in x_d])
                    y = np.array([float(item) for item in y_d])
                    z = np.array([float(item) for item in z_d])

                    value = x.shape[0]
                    id_user = current_user.id

                    print(value)
                    
                    print(latest_hour)

                    response_data = ResponseData(steps=value, timestamp=latest_hour, user_id=id_user)
                    db.session.add(response_data)
                    db.session.commit()


                
            if(len(data_) > 0):
                filename = data_[0]
                hour = filename[0].split(":")[0] + "_" + filename[0].split(":")[1]
                # print("Type hour: ", type(hour))
                formated_hour = hour.replace(" ", "-")

                # print("formated hour: ",formated_hour)
                dir = os.path.join(HOME, formated_hour + ".csv")

                file_object = open(dir, 'a')
                for item in data_:
                    text = item[0] + ";" + str(item[1]) + ";" + str(item[2]) + ";" + str(item[3]) + "\n"
                    file_object.write(text) 
                file_object.close() 



            
            # print(hour)


        #     print(len(data_))
        # global_data.append(data_)
        # data_ = []
    

        # if(len(global_data) == 4):
        #     iter += 1
        #     process_data(global_data, iter)
        #     global_data = []
    
        return str(iter)

def process_data(data, iter):
    lst = []
    for item in data:
        lst.extend(item)
    
    df = pd.DataFrame(lst, columns=(["timestamp", "x", "y", "z"]))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    df.to_csv("/home/hatran_ame/DB_PY/Flask_SQLAlchemy/upload/by_hour/" + str(current_time) + ".csv", sep=";",index=False)
    
    # return(str(iter))

            # submitted_file = request.files['file']
            # if submitted_file and allowed_filename(submitted_file.filename):
            #     filename = secure_filename(
            #         submitted_file.filename)
            #     split_filename = '.' in filename and filename.rsplit('.', 1)
            #     filename = ''.join(
            #         split_filename[:-1]) + "_" + str(time.time()) + "." + split_filename[-1]
            #     os.makedirs(os.path.dirname(UPLOAD_FOLDER), exist_ok=True)
            #     submitted_file.save(os.path.join(UPLOAD_FOLDER, filename))
            # else:
            #     return jsonify({"error": "Exception: {}".format(e)}), 404

            # firstLine = open(os.path.join(UPLOAD_FOLDER, filename)).readline().split(" ")[
            #     3].strip().replace("T", " ")

        #     current_hour = datetime.strptime(firstLine, '%Y-%m-%d %H:%M:%S')

        #     response_data = ResponseData.query.all()
        #     get_one = None
        #     for response in response_data:
        #         ts = response.timestamp
        #         if ts.year == current_hour.year and ts.month == current_hour.month and ts.day == current_hour.day and ts.hour == current_hour.hour:
        #             get_one = response

        #     if get_one == None:
        #         get_one = ResponseData(steps=100, timstamp=current_hour)
        #         db.session.add(get_one)
        #         db.session.commit()
        #     else:
        #         get_one.steps += 1000
        #         get_one.timestamp = current_hour
        #         db.session.commit()

        #     with open(os.path.join(UPLOAD_FOLDER, filename)) as f:
        #         data = f.readlines()
        #         for line in data:

        #             arr = line.split(" ")
        #             x = Decimal(arr[0])
        #             y = Decimal(arr[1])
        #             z = Decimal(arr[2])
        #             time_string = arr[3].strip().replace("T", " ")

        #             timestamp = datetime.strptime(
        #                 time_string, "%Y-%m-%d %H:%M:%S")
        #             acc = Accelerometer(
        #                 x=x, y=y, z=z, timestamp=timestamp, id_user=current_user.id)
        #             db.session.add(acc)
        #             db.session.commit()

        #             res = {
        #                 'msg': get_one.toDict()
        #             }
        #         except Exception as e:
        #             return jsonify({"error": "Exception: {}".format(e)}), 400
        #         return jsonify(res), 200
        # elif request.method == 'GET':
        #     try:
        #         data = Accelerometer.query.filter(
        #             Accelerometer.user_id == current_user.id).group_by(func.date_format(Accelerometer.timestamp, '%H')).all()

        #         res = []
        #         for acc in data:
        #             res.append(acc.toDict())
        #     except Exception as e:
        #         return jsonify({"error": "Exception: {}".format(e)}), 400
        #     return jsonify(res), 200
