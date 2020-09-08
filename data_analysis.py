#------------------------------------------------------------------------------------------------------------------
#   Sample program for EMG data loading and manipulation.
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Read data file
data = np.loadtxt("Izquierda, derecha, cerrado.txt") 
samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]

print('Número de muestras: ', data.shape[0])
print('Número de canales: ', data.shape[1])
print('Duración del registro: ', samps / samp_rate, 'segundos')
print(data);

# Time channel
time = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

# Mark data
mark = data[:, 6]

training_samples = {}
for i in range(0, samps):       
    if mark[i] > 0: 
        print("Marca", mark[i], 'Muestra', i, 'Tiempo', time[i]) 

        if  (mark[i] > 100) and (mark[i] < 200):
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
            training_samples[int(condition_id)].append([iniSamp, i])

print('Rango de muestras con datos de entrenamiento:', training_samples)

# Plot data
start_samp = training_samples[103][2][0];
end_samp = training_samples[103][2][1];

plt.plot(time[start_samp:end_samp], chann1[start_samp:end_samp], label = 'Canal 1')
plt.plot(time[start_samp:end_samp], chann2[start_samp:end_samp], color = 'red', label = 'Canal 2')
plt.xlabel('Tiempo (s)')
plt.ylabel('micro V')
plt.legend()
plt.show()

# Power Spectral Density (PSD) (1 second of training data)
win_size = 256
ini_samp = training_samples[103][2][0]
end_samp = ini_samp + win_size
x = chann1[ini_samp : end_samp]
t = time[ini_samp : end_samp]

plt.plot(t, x)
plt.xlabel('Tiempo (s)')
plt.ylabel('micro V')
plt.show()

power, freq = plt.psd(x, NFFT = win_size, Fs = samp_rate)
plt.clf()

start_freq = next(x for x, val in enumerate(freq) if val >= 4.0);
end_freq = next(x for x, val in enumerate(freq) if val >= 60.0);
print(start_freq, end_freq)

start_index = np.where(freq >= 4.0)[0][0]
end_index = np.where(freq >= 60.0)[0][0]

plt.plot(freq[start_index:end_index], power[start_index:end_index])
plt.xlabel('Hz')
plt.ylabel('Power')
plt.show()
