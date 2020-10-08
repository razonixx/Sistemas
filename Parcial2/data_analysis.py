#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
#from keras.utils import np_utils
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing



def psdCalc(_channel, _win_size, _start_samp, _samp_rate, accumPSDPower, _posturas, _mark):
    current_samp = _start_samp
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp : end_samp]

    power, freq = psd(x, NFFT = _win_size, Fs = _samp_rate)   

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    if _mark is not None:
        _posturas.append(int(mark))

    return power[start_index:end_index]

# Read data file
data = np.loadtxt("Abierto - Cerrado - Normal 1.txt") 
samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]

# Time channel
time = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

# Mark data
mark = data[:, 6]

mark_count = 0

training_samples = {}
for i in range(0, samps):       
    if mark[i] > 0: 
        if  (mark[i] > 100) and (mark[i] < 200):
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
                mark_count += 1
            training_samples[int(condition_id)].append([iniSamp, i])

#print("Training Samples", training_samples)

accumPSDPower = []
posturas = []
time_meausred = 256 #256 = 1 sec

for mark in training_samples:
    for window in training_samples[mark]:
        current_window = window[0]
        end_window = window[1]
        while (current_window + time_meausred) < end_window:
            current_window = current_window + time_meausred
            chan1 = psdCalc(chann1, time_meausred, current_window, samp_rate, accumPSDPower, posturas, mark) # 256=1 sec
            chan2 = psdCalc(chann2, time_meausred, current_window, samp_rate, accumPSDPower, None, None) # 256=1 sec
            row = np.append(chan1, chan2)
            accumPSDPower.append(row)

accumPSDPower = np.array(accumPSDPower)
print(accumPSDPower.shape)

x = accumPSDPower
y = np.array(posturas)
lab_enc = preprocessing.LabelEncoder()
y = lab_enc.fit_transform(y)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True)
clf = svm.SVC(kernel = 'linear')

acc = 0
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
