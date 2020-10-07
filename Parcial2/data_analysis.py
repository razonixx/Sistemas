#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from sklearn import svm

def psdCalc(_channel, _start_samp, _end_samp, _samp_rate, accumPSDPower, accumPSDFreq):
    x = _channel[_start_samp : _end_samp]

    power, freq = psd(x, NFFT = (_end_samp - _start_samp), Fs = _samp_rate)   

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    accumPSDPower.append(power[start_index:end_index])
    accumPSDFreq.append(freq[start_index:end_index])

# Read data file
data = np.loadtxt("../Abierto, cerrado, descanso.txt") 
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

#print(training_samples)

accumPSDPowerChan1 = []
accumPSDFreq = []

for mark in training_samples:
    for window in training_samples[mark]:
        print(window)
        psdCalc(chann1, window[0], window[1], samp_rate, accumPSDPowerChan1, accumPSDFreq)
        print(len(accumPSDPowerChan1[0]))
        exit()


'''
# Train SVM classifier with all the available observations
clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

# 5-fold cross-validation
kf = KFold(n_splits=10, shuffle = True)
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
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    

    acc += acc_i 
'''