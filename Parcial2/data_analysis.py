#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

def timeGraph(_start_samp, _end_samp, _channel, _time, pltTitle, pltInstance):
    pltInstance.plot(_time[_start_samp:_end_samp], _channel[_start_samp:_end_samp])
    pltInstance.set_xlabel('Tiempo (s)')
    pltInstance.set_ylabel('micro V')
    pltInstance.set_title(pltTitle)

def psdGraph(_channel, _win_size, _start_samp, _samp_rate, pltTitle, pltInstance):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp : end_samp]

    power, freq = psd(x, NFFT = _win_size, Fs = _samp_rate)   

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    pltInstance.plot(freq[start_index:end_index], power[start_index:end_index])
    pltInstance.set_xlabel('Hz')
    pltInstance.set_ylabel('Power')
    pltInstance.set_title(pltTitle)

def fourGraphs(_training_samples, _channel1, _channel2, _time, _samp_rate, _psd_win_size, _mark, _window):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)

    timeGraph(_training_samples[_mark][_window][0], _training_samples[_mark][_window][1], _channel1, _time, "Mark: " + str(_mark) + ", Channel: 1, Window: " + str(_window+1), axes[0, 0])
    timeGraph(_training_samples[_mark][_window][0], _training_samples[_mark][_window][1], _channel2, _time, "Mark: " + str(_mark) + ", Channel: 2, Window: " + str(_window+1), axes[0, 1])
    psdGraph(_channel1, _psd_win_size, _training_samples[_mark][_window][0], _samp_rate, "PSD", axes[1, 0])
    psdGraph(_channel2, _psd_win_size, _training_samples[_mark][_window][0], _samp_rate, "PSD", axes[1, 1])

    fig.tight_layout()
    plt.show()

def psdCalc(_channel, _win_size, _start_samp, _samp_rate, accumPSDPower, accumPSDFreq):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp : end_samp]

    power, freq = psd(x, NFFT = _win_size, Fs = _samp_rate)   

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    accumPSDPower.append(power[start_index:end_index])
    accumPSDFreq.append(freq[start_index:end_index])

def psdCalcAccum(_training_samples, _channel1, _channel2, _samp_rate, accumPSDFreq, finalPSDPowerChan1, finalPSDPowerChan2):
    accumPSDPowerChan1 = []
    accumPSDPowerChan2 = []
    tempPSDPowerChan1 = {}
    tempPSDPowerChan2 = {}

    for mark in _training_samples:
        if not mark in finalPSDPowerChan1.keys():
                tempPSDPowerChan1[mark] = []
                tempPSDPowerChan2[mark] = []
                finalPSDPowerChan1[mark] = [0] * 56
                finalPSDPowerChan2[mark] = [0] * 56
        for window in _training_samples[mark]:
            psdCalc(_channel1, 256, window[0], samp_rate, accumPSDPowerChan1, accumPSDFreq)
            psdCalc(_channel2, 256, window[0], samp_rate, accumPSDPowerChan2, accumPSDFreq)
        tempPSDPowerChan1[mark] = accumPSDPowerChan1
        tempPSDPowerChan2[mark] = accumPSDPowerChan2
        accumPSDPowerChan1 = []
        accumPSDPowerChan2 = []

    for mark in finalPSDPowerChan1:
        for i in range(0, len(tempPSDPowerChan1[mark])):
            for j in range(0, len(tempPSDPowerChan1[mark][i])):
                finalPSDPowerChan1[mark][j] = finalPSDPowerChan1[mark][j] + tempPSDPowerChan1[mark][i][j]
                finalPSDPowerChan2[mark][j] = finalPSDPowerChan2[mark][j] + tempPSDPowerChan2[mark][i][j]

    for mark in finalPSDPowerChan1:
        for i in range(0, len(finalPSDPowerChan1[mark])):
            finalPSDPowerChan1[mark][i] = (finalPSDPowerChan1[mark][i])/len(tempPSDPowerChan1[101][0])
            finalPSDPowerChan2[mark][i] = finalPSDPowerChan2[mark][i]/len(tempPSDPowerChan2[101][0])
    
# Graficar 3 psd, 1 por cada marca
def graphsAccum(_training_samples, _channel1, _channel2, _time, _samp_rate, _mark_count):
    fig, axes = plt.subplots(nrows=2, ncols=_mark_count, sharex=False, sharey=False)
    accumPSDFreq = []
    finalPSDPowerChan1 = {}
    finalPSDPowerChan2 = {}
    mark_index = 0

    psdCalcAccum(training_samples, chann1, chann2, samp_rate, accumPSDFreq, finalPSDPowerChan1, finalPSDPowerChan2)

    #print(finalPSDPowerChan1[101])
    #exit()

    for key_mark in finalPSDPowerChan1:
        axes[0,mark_index].plot(accumPSDFreq[0], finalPSDPowerChan1[key_mark])
        axes[0,mark_index].set_xlabel('Hz')
        axes[0,mark_index].set_ylabel('Power')
        axes[0,mark_index].set_title('Mark ' + str(int(key_mark)) + ' Channel 1')

        axes[1,mark_index].plot(accumPSDFreq[0], finalPSDPowerChan2[key_mark])
        axes[1,mark_index].set_xlabel('Hz')
        axes[1,mark_index].set_ylabel('Power')
        axes[1,mark_index].set_title('Mark ' + str(int(key_mark)) + ' Channel 2')

        mark_index+=1

    fig.tight_layout()
    plt.show()

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

fourGraphs(training_samples, chann1, chann2, time, 256, 256, 101, 4)
fourGraphs(training_samples, chann1, chann2, time, 256, 256, 102, 2)
#fourGraphs(training_samples, chann1, chann2, time, 256, 256, 103, 5)

graphsAccum(training_samples, chann1, chann2, time, samp_rate, mark_count)


