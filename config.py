# class Config():
DEBUG = True
SECRET_KEY = 'HA_ACCELEROMETER'

APP_HOST = '0.0.0.0'
APP_PORT = 8080
APP_SELF_REF = 'accelerometer'
SQLALCHEMY_DATABASE_URI = 'mysql://root:Long.0311@127.0.0.1:3306/acc_data'

SQLALCHEMY_TRACK_MODIFICATIONS = False
API_URI = f'http://{APP_HOST}:{APP_PORT}'
URL = f'http://{APP_HOST}:{APP_PORT}'
API_URI_SR = f'http://{APP_SELF_REF}:{APP_PORT}'
UPLOAD_FOLDER = 'upload/data/'
