from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from keras import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.utils import np_utils
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class MegaClassifier:
    def __init__(self, vector_size=56, output_size=3):

        self.mlp = Sequential()

        self.vector_size = vector_size

        self.mlp.add(Dense(
            128, activation='relu', kernel_initializer='random_normal', input_dim=self.vector_size))
        self.mlp.add(Dense(128, activation='sigmoid'))
        self.mlp.add(Dense(output_size, activation='softmax'))
        self.mlp.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.models = {
            'MLP': self.mlp
        }

    def add(self, model):
        self.models.append(model)

    def train(self, X, Y, nn_Y):
        self.mlp.fit(X, nn_Y, epochs=100, batch_size=10, verbose=0)

    def test(self, x):

        y_hat = {}
        for name, model in self.models.items():
            y_hat[name] = model.predict(x)

        return y_hat


if __name__ == '__main__':
    print('lmao')
