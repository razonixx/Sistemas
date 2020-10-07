# ------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
# ------------------------------------------------------------------------------------------------------------------

import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import matplotlib.animation as animation
import matplotlib
from time import time as time_fn
matplotlib.use('Qt4Agg')


def timeGraph(_start_samp, _end_samp, _channel, _time, pltTitle, pltInstance):
    line, = pltInstance.plot(_time[_start_samp:_end_samp],
                             _channel[_start_samp:_end_samp])
    pltInstance.set_xlabel('Tiempo (s)')
    pltInstance.set_ylabel('micro V')
    pltInstance.set_title(pltTitle)
    return line


def psdGraph(_channel, _win_size, _start_samp, _samp_rate, pltTitle, pltInstance):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp: end_samp]

    power, freq = psd(x, NFFT=_win_size, Fs=_samp_rate)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    line, = pltInstance.plot(
        freq[start_index:end_index], power[start_index:end_index])
    pltInstance.set_xlabel('Hz')
    pltInstance.set_ylabel('Power')
    pltInstance.set_title(pltTitle)
    return line


def fourGraphs(_training_samples, _channel1, _channel2, _time, _samp_rate, _psd_win_size, _mark, _window):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)

    line0 = timeGraph(_training_samples[_mark][_window][0], _training_samples[_mark][_window][1], _channel1,
                      _time, "Mark: " + str(_mark) + ", Channel: 1, Window: " + str(_window+1), axes[0, 0])
    line1 = timeGraph(_training_samples[_mark][_window][0], _training_samples[_mark][_window][1], _channel2,
                      _time, "Mark: " + str(_mark) + ", Channel: 2, Window: " + str(_window+1), axes[0, 1])
    line2 = psdGraph(_channel1, _psd_win_size,
                     _training_samples[_mark][_window][0], _samp_rate, "PSD", axes[1, 0])
    line3 = psdGraph(_channel2, _psd_win_size,
                     _training_samples[_mark][_window][0], _samp_rate, "PSD", axes[1, 1])

    fig.tight_layout()
    return fig, axes, [line0, line1, line2, line3]


def psdCalc(_channel, _win_size, _start_samp, _samp_rate, accumPSDPower, accumPSDFreq):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp: end_samp]

    power, freq = psd(x, NFFT=_win_size, Fs=_samp_rate)

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
            psdCalc(_channel1, 256,
                    window[0], samp_rate, accumPSDPowerChan1, accumPSDFreq)
            psdCalc(_channel2, 256,
                    window[0], samp_rate, accumPSDPowerChan2, accumPSDFreq)
        tempPSDPowerChan1[mark] = accumPSDPowerChan1
        tempPSDPowerChan2[mark] = accumPSDPowerChan2
        accumPSDPowerChan1 = []
        accumPSDPowerChan2 = []

    for mark in finalPSDPowerChan1:
        for i in range(0, len(tempPSDPowerChan1[mark])):
            for j in range(0, len(tempPSDPowerChan1[mark][i])):
                finalPSDPowerChan1[mark][j] = finalPSDPowerChan1[mark][j] + \
                    tempPSDPowerChan1[mark][i][j]
                finalPSDPowerChan2[mark][j] = finalPSDPowerChan2[mark][j] + \
                    tempPSDPowerChan2[mark][i][j]

    for mark in finalPSDPowerChan1:
        for i in range(0, len(finalPSDPowerChan1[mark])):
            finalPSDPowerChan1[mark][i] = (
                finalPSDPowerChan1[mark][i])/len(tempPSDPowerChan1[101][0])
            finalPSDPowerChan2[mark][i] = finalPSDPowerChan2[mark][i] / \
                len(tempPSDPowerChan2[101][0])

# Graficar 3 psd, 1 por cada marca


def graphsAccum(_training_samples, _channel1, _channel2, _time, _samp_rate, _mark_count):
    fig, axes = plt.subplots(nrows=2, ncols=_mark_count,
                             sharex=False, sharey=False)
    accumPSDFreq = []
    finalPSDPowerChan1 = {}
    finalPSDPowerChan2 = {}
    mark_index = 0

    psdCalcAccum(training_samples, chann1, chann2, samp_rate,
                 accumPSDFreq, finalPSDPowerChan1, finalPSDPowerChan2)

    # print(finalPSDPowerChan1[101])
    # exit()

    for key_mark in finalPSDPowerChan1:
        axes[0, mark_index].plot(accumPSDFreq[0], finalPSDPowerChan1[key_mark])
        axes[0, mark_index].set_xlabel('Hz')
        axes[0, mark_index].set_ylabel('Power')
        axes[0, mark_index].set_title(
            'Mark ' + str(int(key_mark)) + ' Channel 1')

        axes[1, mark_index].plot(accumPSDFreq[0], finalPSDPowerChan2[key_mark])
        axes[1, mark_index].set_xlabel('Hz')
        axes[1, mark_index].set_ylabel('Power')
        axes[1, mark_index].set_title(
            'Mark ' + str(int(key_mark)) + ' Channel 2')

        mark_index += 1

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
        if (mark[i] > 100) and (mark[i] < 200):
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
                mark_count += 1
            training_samples[int(condition_id)].append([iniSamp, i])

fig, axes, lines = fourGraphs(training_samples, chann1,
                              chann2, time, 256, 256, 101, 4)

line0, line1, line2, line3 = lines


def update(data):
    if data is None:
        return axes
    line0.set_data(
        data[0][0],
        data[0][1]
    )
    line1.set_data(
        data[1][0],
        data[1][1]
    )
    line2.set_data(
        data[2][0],
        data[2][1]
    )
    line3.set_data(
        data[3][0],
        data[3][1]
    )

    for i in range(2):
        for j in range(2):
            axes[i][j].relim()
            axes[i][j].autoscale_view()

    return axes


# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

start_time = time_fn()


def data_gen():
    n_channels = 6
    samp_rate = 256
    emg_data = [[] for i in range(n_channels)]
    samp_count = 0
    while True:
        try:
            data, addr = sock.recvfrom(1024*1024)

            values = np.frombuffer(data)
            ns = int(len(values)/n_channels)
            samp_count += ns

            for i in range(ns):
                for j in range(n_channels):
                    emg_data[j].append(values[n_channels*i + j])

            elapsed_time = time_fn() - start_time
            # CHANNEL 1 / 2
            time_x = []
            c = 0.1 / 25
            for _ in range(25):
                time_x.append(elapsed_time)
                elapsed_time += c

            # PSD
            power1, freq1 = psd(emg_data[0][-25:], NFFT=256, Fs=256)
            power2, freq2 = psd(emg_data[2][-25:], NFFT=256, Fs=256)

            start_index = np.where(freq1 >= 4.0)[0][0]
            end_index = np.where(freq1 >= 60.0)[0][0]
            data = [
                [time_x[-25:], emg_data[0][-25:]],
                [time_x[-25:], emg_data[2][-25:]],
                [freq1[start_index:end_index], power1[start_index:end_index]],
                [freq2[start_index:end_index], power2[start_index:end_index]],

            ]
            yield data

        except socket.timeout:
            yield None


ani = animation.FuncAnimation(fig, update, data_gen, interval=0,
                              save_count=50)
plt.show()
#fourGraphs(training_samples, chann1, chann2, time, 256, 256, 101, 4)
#fourGraphs(training_samples, chann1, chann2, time, 256, 256, 102, 2)
#fourGraphs(training_samples, chann1, chann2, time, 256, 256, 103, 5)

#graphsAccum(training_samples, chann1, chann2, time, samp_rate, mark_count)
