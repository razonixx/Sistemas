import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from pprint import pprint
import tensorflow as tf
from models import MegaClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from glob import glob

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def psdCalc(_channel, _win_size, _start_samp, _samp_rate, accumPSDPower, _posturas, _mark):
    end_samp = _start_samp + _win_size

    x = _channel[_start_samp: end_samp]

    power, freq = psd(x, NFFT=_win_size, Fs=_samp_rate)

    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    if _mark is not None:
        _posturas.append(int(_mark))

    return power[start_index:end_index]


def data_loader(file_name = "Abierto - Cerrado - Normal 1.txt"):
    # Read data file
    data = np.loadtxt(file_name)
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

    #print("Training Samples", training_samples)

    accumPSDPower = []
    posturas = []
    time_meausred = 256  # 256 = 1 sec

    for mark in training_samples:
        for window in training_samples[mark]:
            current_window = window[0]
            end_window = window[1]
            while (current_window + time_meausred) < end_window:
                current_window = current_window + time_meausred
                chan1 = psdCalc(chann1, time_meausred, current_window,
                                samp_rate, accumPSDPower, posturas, mark)  # 256=1 sec
                chan2 = psdCalc(chann2, time_meausred, current_window,
                                samp_rate, accumPSDPower, None, None)  # 256=1 sec
                row = np.append(chan1, chan2)
                accumPSDPower.append(row)

    accumPSDPower = np.array(accumPSDPower)
    x = accumPSDPower
    y = np.array(posturas)
    return x,y, mark_count


data = {}
for file in glob('*.txt'):
    data[file[:-4]] = data_loader(file)

def run_model(train_data, test_data):
    x,y, count = train_data
    x_test, y_test, count_test = test_data
    classifier = MegaClassifier(vector_size=112, output_size=count)
    encoder = LabelEncoder()
    
    encoder.fit(y)

    y_test = encoder.transform(y_test)
    y = encoder.transform(y)

    # convert integers to dummy variables (i.e. one hot encoded)
    nn_y = np_utils.to_categorical(y)

    N_SPLITS = 5

    kf = KFold(n_splits=N_SPLITS, shuffle = True)

    acc_results = {}

    for train_index, test_index in kf.split(x):
        # Training phase
        x_train = x[train_index, :]
        y_train = y[train_index]
        nn_y_train = nn_y[train_index]

        classifier.train(x_train,y_train, nn_y_train)

        # Test phase
        x_validation = x[test_index, :]
        y_validation = y[test_index]
        nn_y_validation = nn_y[test_index]

        y_predict = classifier.test(x_validation)
        for name, y_hat in y_predict.items():
            if name in ('MLP', 'Perceptron'):
                y_hat = list(map(np.argmax,y_hat))

            acc = accuracy_score(y_validation, y_hat)
            acc_results[name] = acc_results.get(name, 0) + acc

    for name, acc in acc_results.items():
        acc_results[name] = acc / N_SPLITS
    
    
    test_results = {}
    y_predict = classifier.test(x_test)
    for name, y_hat in y_predict.items():
            if name in ('MLP', 'Perceptron'):
                y_hat = list(map(np.argmax,y_hat))

            acc = accuracy_score(y_test, y_hat)
            test_results[name] = test_results.get(name, 0) + acc
    
    return {'training': acc_results, 'test': test_results, 'classifier': classifier, 'encoder': encoder}

acum_res = {}

files = [
    ('Abierto - Cerrado - Normal 1','Abierto - Cerrado - Normal 2'),
    ('Izquierda - Derecha - Cerrado 1', 'Izquierda - Derecha - Cerrado 2'),
    ('Abierto - Cerrado 1','Abierto - Cerrado 2'),
    ('Abierto - Cerrado 1', 'Abierto - Cerrado 3')
]

classifiers = {}
encoders = {}

for train_file, test_file in files:
    results = run_model(data[train_file], data[test_file])
    classifiers[train_file] = results['classifier']
    encoders[train_file] = results['encoder']
    print(train_file,end='\n\n')
    for model, score in results['test'].items():
        acum_res[train_file] = acum_res.get(train_file,0) + score 
        print(f'{model} | {round(score * 100, 4)}')
    print('---------')