import string
from struct import unpack
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
import random
import sys
import joblib
from numpy import NaN, Inf, arange, isscalar, asarray, array
import sklearn
import pickle
import pandas as pd
from cmath import sqrt
import numpy as np
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
plt.rcParams["figure.figsize"] = (30, 20)  # Thay chinh size o day cung duoc a
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15, loc='lower right')
import hydra
from hydra import utils
import numpy as np
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import os

with initialize(config_path="configs"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)


class Signal_new:

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.a = None
        self.all_index_peaks = 0
        self.fs = None
        self.a_lpf = None
        self.thr = None
        self.a_pulse = None
        self.x_f = None
        self.y_f = None
        self.z_f = None
        self.fs = 100
        self.x_f = None
        self.y_f = None
        self.z_f = None
    

    def set_signal(self, signal, isFilter=True, cutoff=5.0, order=5, fs=100):
        # FOR DATA FROM ANDROID LOADED FILE
        # timestamp, x, y, z = np.genfromtxt(str(path), delimiter=";", dtype='str',unpack=True)
        # self.x = np.array([float(item) for item in x])
        # self.y = np.array([float(item) for item in y])
        # self.z = np.array([float(item) for item in z])

        # FOR DATAS FROM IPHONE
        # _,x,y,z = 10*np.loadtxt(path, delimiter=";", skiprows=1, unpack=True)
        # self.x = np.array(x)
        # self.y = np.array(y)
        # self.z = np.array(z)

        self.x = np.array([item[0] for item in signal])
        self.y = np.array([item[1] for item in signal])
        self.z = np.array([item[2] for item in signal])


        self.a = np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        if isFilter:
            self.low_pass_filter(cutoff=cutoff, order=order)

    def butter(self, cutoff, order):
        f_nyq = 0.5 * self.fs
        f_normal = cutoff / f_nyq
        numerator, denominator = butter(
            order, f_normal, btype='low', analog=False)
        return numerator, denominator

    def butter_lowpass_filter(self, data, cutoff=5.0, order=5):
        b, a = self.butter(cutoff, order=order)
        y = filtfilt(b, a, data)
        return y

    def low_pass_filter(self, cutoff=5.0, order=5):

        self.a_lpf = self.butter_lowpass_filter(
            self.a, cutoff=cutoff, order=order)
        self.x_f = self.butter_lowpass_filter(
            self.x, cutoff=cutoff, order=order)
        self.y_f = self.butter_lowpass_filter(
            self.y, cutoff=cutoff, order=order)
        self.z_f = self.butter_lowpass_filter(
            self.z, cutoff=cutoff, order=order)
        self.thr = np.average(self.a_lpf)

    def collect_index_peaks(self, sig_pulse, num_thr=10):
        list_index = []
        peaks, _ = find_peaks(sig_pulse)
        for idx, peak in enumerate(peaks):
            indices = []
            for i, e in enumerate(reversed(range(0, peak, 1))):

                if sig_pulse[e] == 0:
                    break
                indices.append(e)
            # indices                                                                                                                                                                           = indices[::-1]
            indices.reverse()
            for i, e in enumerate(range(peak, len(sig_pulse), 1)):

                if sig_pulse[e] == 0:
                    break
                indices.append(e)
            if len(indices) > num_thr:
                list_index.append(indices)
            else:
                peaks = np.delete(peaks, i)
        return list_index

    def create_square_pulse(self, sig, thr):
        # create a numpy array to store binary pulses
        sig_pulse = np.zeros_like(sig)

        for i, e in enumerate(sig):
            if e >= thr:
                sig_pulse[i] = 1
        return sig_pulse

    def get_all_peaks(self, num_thr=5, ratio=1.03):
        self.a_pulse = self.create_square_pulse(self.a_lpf, ratio*self.thr)
        # print(self.a_pulse[:200])
        list_indexs = self.collect_index_peaks(self.a_pulse, num_thr=num_thr)

        features = []
        for i in list_indexs:
            features.append(self.a_lpf[i])
        features = np.array(features)

        index_peaks = []
        for i, e in enumerate(features):
            idx = np.where(e == max(e))[0][0]
            index_peaks.append(list_indexs[i][idx])

        self.all_index_peaks = np.array(index_peaks)
        return self.all_index_peaks


    # Fit n_point = window_size
    def scale_data_point(self, peak, sig, n_point):
        data_points = []
        left_peaks = []
        right_peaks = []
        id_peaks = []
        n = n_point//2
        if peak <= n:
            left_peaks = sig[:peak]
            right_peaks = sig[peak:n_point]
            index_peaks = range(n_point)
        elif peak+n > len(sig):
            right_peaks = sig[peak:len(sig)]
            num = n_point-(len(sig)-peak)
            left_peaks = sig[peak-num:peak]
            index_peaks = range(peak-num, len(sig), 1)

        else:
            left_peaks = sig[peak-n:peak]
            right_peaks = sig[peak:peak+n]
            index_peaks = range(peak-n, peak+n, 1)

        data_points = np.concatenate((left_peaks, right_peaks))
        return data_points, index_peaks

    def get_window_data(self, peak, x, y, z, a, a_lpf, num_point):
        data_points_a_lpf, index_window = self.scale_data_point(
            peak, a_lpf, num_point)
        data_points_x = x[index_window]
        data_points_y = y[index_window]
        data_points_z = z[index_window]
        data_points_a = a[index_window]
        # print("data point x: " + str(len(data_points_x)))

        h = a_lpf[peak]
        data = []
        data.append(data_points_x)
        data.append(data_points_y)
        data.append(data_points_z)
        data.append(data_points_a)

        data.append(data_points_a_lpf)
        data.append(h)

        return data

    def resample(self, sig, num):
        return signal.resample(sig, num)

    def fft_transform(self, sig):
        dt = 1.0/self.fs
        n = len(sig)
        fhat = np.fft.fft(sig, n)
        PSD = fhat*np.conj(fhat)/n  # phổ công suất
        freq = (1/(dt*n)) * np.arange(n)
        L = np.arange(1, np.floor(n/2), dtype='int')
        return fhat, PSD, freq, L

    def get_psd_values(self, sig):
        _, PSD, freq, L = self.fft_transform(sig)
        psd = np.abs(PSD[L])
        return psd

    def create_data_point(self, data, thr, num_feature_psd=15):
        #         time domain
        dt = []
        mean = np.mean(data[3])
        std = np.std(data[3])
        max_ = max(data[3])
#         min_ = min(data[4])
        dt.append(mean)
        dt.append(std)
        dt.append(max_)
#         dt.append(min_)

#         freq domain
        psd_x = self.get_psd_values(data[0])
        psd_y = self.get_psd_values(data[1])
        psd_z = self.get_psd_values(data[2])
        psd_a = self.get_psd_values(data[4])

        data_points = []
        data_points.extend(dt)
        data_points.extend(data[4])
        data_points.extend(psd_x[:num_feature_psd])
        data_points.extend(psd_y[:num_feature_psd])
        data_points.extend(psd_z[:num_feature_psd])
        data_points.extend(psd_a[:num_feature_psd])
        return data_points

    def get_data_point(self, peak, num_point, num_resample=64):
        data = self.get_window_data(
            peak, self.x_f, self.y_f, self.z_f, self.a, self.a_lpf, num_point)
        # print(data)
        if num_point != num_resample:
            data[0] = self.resample(data[0], num_resample)
            data[1] = self.resample(data[1], num_resample)
            data[2] = self.resample(data[2], num_resample)
            data[3] = self.resample(data[3], num_resample)
            data[4] = self.resample(data[4], num_resample)
        # print(len(data[0]))
        data_point = self.create_data_point(data, self.thr)
        return data_point

    def visual_predict(self, y_predict, isSave=False, filename='default.eps'):
        y_true = []
        y_false = []
        for i, e in enumerate(y_predict):
            if e != 0:
                y_true.append(self.all_index_peaks[i])
            else:
                y_false.append(self.all_index_peaks[i])

        y_true = np.array(y_true)
        y_false = np.array(y_false)

        text = "Total peaks: " + str(y_true.shape[0])
        # x_axis = (len(self.a_lpf) / 2 - 30*5)

        # y_axis = np.max(self.a_lpf) + 0.5
        # print(x_axis)

        plt.figure(1, figsize=(30, 10))
        plt.plot(self.a_lpf, 'olive')

        if y_true.shape[0] != 0:
            plt.plot(y_true, self.a_lpf[y_true], 'o',
                     color="forestgreen", label='Step Detect')
        if y_false.shape[0] != 0:
            plt.plot(y_false, self.a_lpf[y_false], 'xr', label='Fake Peak')

        plt.legend()
        plt.title(text, fontsize=16, color='purple')
        plt.xlabel("Samples by time")
        plt.ylabel("Amplitude")
        # plt.text(x_axis, y_axis, text, fontsize=16, color='r')
        plt.show()
        if isSave:
            plt.savefig(filename)
        print(y_true.shape[0])


class StepPredict:
    def __init__(self, window_size):
        self.model = None
        self.scaler = None
        self.x_test = None
        self.y_test = None
        self.window_size = window_size

    def load_model(self, path_file):
        self.model = pickle.load(open(path_file, 'rb'))

    def loadScaler(self, path_file):
        self.scaler = pickle.load(open(path_file, 'rb'))

    def predict_peak(self, data_point):
        data_point = np.array(data_point)
        # print(data_point.shape)
        data_point = data_point.reshape(1, -1)
        data_point_norm = self.scaler.transform(data_point)
        y_predict = self.model.predict(data_point_norm)
        return y_predict[0]

    def process(self, signal):
        peaks = signal.all_index_peaks
        print(peaks)
        pre_window = self.window_size[0]
        y_predicts = []
        y_window = []
        pre_peak = 0
        distance_peak = 10000000
        T_thr1 = 32
        T_thr2 = 21

        for peak in peaks:
            num_size = pre_window
            data_point = signal.get_data_point(peak, num_size)
            y_pre = self.predict_peak(data_point)
            if y_pre == 0:
                for size in self.window_size:
                    if size != pre_window:
                        data_point = signal.get_data_point(peak, size)
                        y_pre = self.predict_peak(data_point)
                        if y_pre != 0:
                            pre_window = size
                            break

            #   nguong loai diem canh nhau
            if (y_pre != 0):
                distance_peak = peak-pre_peak
                if (pre_window == max(self.window_size)):
                    if distance_peak < T_thr1:
                        y_pre = 0
                else:
                    if distance_peak <= T_thr2:
                        y_pre = 0
                pre_peak = peak
            y_predicts.append(y_pre)
            y_window.append(pre_window)
        return y_predicts, y_window

def scale_data(X_train, X_test):
    sc = StandardScaler()
    sc_X_train = sc.fit_transform(X_train)
    sc_X_test = sc.transform(X_test)
    return (sc_X_train, sc_X_test)

def train_test_model(model, train_X, train_y, test_X, test_y):
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    acc = accuracy_score(test_y, prediction)
    return prediction, acc, model
    
def classifiers_trials(cls, train_X, train_y, test_X, test_y):
    log_cols=["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)
    for cl in cls:
        pred, accuracy, model = train_test_model(cl, train_X, train_y, test_X, test_y)
        # cl.fit(train_X, train_y)
        name = cl.__class__.__name__
        print("="*30)
        print(name)
        print('****Results****')
        # prediction = cl.predict(test_X)
        # accuracy = accuracy_score(test_y, prediction)
        print("Accuracy: {:.4%}".format(accuracy))
        log_entry = pd.DataFrame([[name, accuracy*100]], columns=log_cols)
        log = log.append(log_entry)
        joblib.dump(model, open("/home/hatran_ame/DATN/step_detection_upgrade/data/data_ha/processed_data/data_20_7/trials/" + name + ".model", 'wb'))
    print("="*30)
    return log


def predict(path):
    sig = Signal_new()

    sig.set_signal(path)

    #initialize StepDetection module with list of window_size
    stepDt = StepPredict(window_size=[64, 48, 32])

    model_path = data_cfg.processed_data.data_30_7.model
    scaler_path = data_cfg.processed_data.data_30_7.scaler

    peaks = sig.get_all_peaks()

    stepDt.load_model(model_path)
    stepDt.loadScaler(scaler_path)

    #adaptive resampling
    y, w = stepDt.process(sig)

    true_peak = []
    for i, e in enumerate(peaks):
        if y[i] != 0:
            true_peak.append(e)
    true_peak = np.array(true_peak)
    return true_peak.shape[0]


def convert_data(data: list) -> list:
    lst = []
    for i, e in enumerate(data):
        lst.append([data[i]['valueX'], data[i]['valueY'], data[i]['valueZ']])

    return lst

# Test import lib and package
# import warnings
# import numpy as np
# from modules.collect_data import read_csv
# import hydra
# from hydra import utils
# import random

# def get_abs_path(file_name):
#     return utils.to_absolute_path(file_name)

# # Check hydra path
# @hydra.main(config_path='../../configs',
#             config_name='data_path')


def main():
    # timestamp, x, y, z = np.genfromtxt(
    #     "/home/hatran_ame/DATN/step_detection_upgrade/src/code_ha/outputs/2.csv", delimiter=";", dtype='str',unpack=True)
    # x_d = x[1:]
    # y_d = y[1:]
    # z_d = z[1:]
    # x_data = [float(item) for item in x_d]
    # y_data = [float(item) for item in y_d]
    # z_data = [float(item) for item in z_d]

    # print(x)
    # sig = Signal_new()
    # sig.set_signal("/home/hatran_ame/DATN/step_detection_upgrade/src/code_ha/outputs/1.csv")
    # sig.visual_predict
    # print(predict(signal).shape[0])

    predict("/home/hatran_ame/DB_PY/Flask_SQLAlchemy/upload/by_hour/2022-07-27-13_22.csv")

if __name__ == "__main__":
    main()
