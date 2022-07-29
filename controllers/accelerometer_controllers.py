from re import T
from flask_cors import CORS
import logging
from Predict import StepPredict, Signal_new, predict

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
            # iter += 1
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
            # glob

            # path of the directory
            path = "upload/by_hour"
            
            # Getting the list of directories
            dir = os.listdir(path)
            
            # Checking if the list is empty or not
            if len(dir) != 0:
        
                list_of_files = glob.glob('upload/by_hour/*') # * means all if need specific format then *.csv
                latest_file = max(list_of_files, key=os.path.getctime)

                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d-%H")

                latest_info = latest_file.split(".csv")[0].split(HOME)[1] 
                latest_hour = latest_info.split("_")[0]
                print(latest_hour)
                id_user = current_user.id
                

############################HERE=============================================
                if latest_file != str(current_time) + "_" + str(id_user):
                    data = ResponseData.query.filter(ResponseData.user_id==current_user.id).all()
                    print("latest_info", latest_info)
                    print("current time", str(current_time) + "_" + str(id_user))


                    print("check: ", latest_info == str(current_time) +"_"+  str(id_user))
                    check = False
                    for i in data:
                        temp = str(i.timestamp).split(":")[0]
                        if(temp == latest_hour):
                            check = True
                            break
                    if(check == False):
                        timestamp, x, y, z = np.genfromtxt(str(latest_file), delimiter=";", dtype='str',unpack=True)
                        x_d = x[1:]
                        y_d = y[1:]
                        z_d = z[1:]
                        x = np.array([float(item) for item in x_d])
                        y = np.array([float(item) for item in y_d])
                        z = np.array([float(item) for item in z_d])

                        value = x.shape[0]

                        steps = predict(latest_file)

                        print("step: ", steps)

                        response_data = ResponseData(steps=steps, timestamp=latest_hour, user_id=id_user)
                        db.session.add(response_data)
                        db.session.commit()


            print("[Run here]")
            if(len(data_) > 0):
                filename = data_[0]
                hour = filename[0].split(":")[0]
                print("hour ", hour)
                # print("Type hour: ", type(hour))
                formated_file = hour.replace(" ", "-") + "_" + str(id_user)

                # print("formated hour: ",formated_hour)
                dir = os.path.join(HOME, formated_file + ".csv")

                file_object = open(dir, 'a')
                for item in data_:
                    text = item[0] + ";" + str(item[1]) + ";" + str(item[2]) + ";" + str(item[3]) + "\n"
                    file_object.write(text) 
                file_object.close() 
    
        return str("Uploadeds")

# def process_data(data, iter):
#     lst = []
#     for item in data:
#         lst.extend(item)
    
#     df = pd.DataFrame(lst, columns=(["timestamp", "x", "y", "z"]))

#     now = datetime.now()
#     current_time = now.strftime("%H:%M:%S")
#     df.to_csv("/home/hatran_ame/DB_PY/Flask_SQLAlchemy/upload/by_hour/" + str(current_time) + ".csv", sep=";",index=False)
    