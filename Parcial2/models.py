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

        self.linear_svc = svm.SVC(kernel='linear')
        self.radial_svc = svm.SVC(kernel='rbf')
        self.knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        self.tree = tree.DecisionTreeClassifier()
        self.perceptron = Sequential()
        self.mlp = Sequential()

        self.vector_size = vector_size

        self.perceptron.add(Dense(
            128, activation='relu', kernel_initializer='random_normal', input_dim=self.vector_size))
        self.perceptron.add(Dense(output_size, activation='softmax'))
        self.perceptron.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.mlp.add(Dense(
            128, activation='relu', kernel_initializer='random_normal', input_dim=self.vector_size))
        self.mlp.add(Dense(128, activation='sigmoid'))
        self.mlp.add(Dense(output_size, activation='softmax'))
        self.mlp.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.models = {
            'SVD_Lineal': self.linear_svc,
            'SVD_Radial': self.radial_svc,
            'KNN': self.knn,
            'Tree': self.tree,
            'Perceptron': self.perceptron,
            'MLP': self.mlp
        }

    def add(self, model):
        self.models.append(model)

    def train(self, X, Y, nn_Y):
        #print("Training: SVD linear...")
        self.linear_svc.fit(X, Y)
        #print("Training: SVD Radial...")
        self.radial_svc.fit(X, Y)
        #print("Training: SVD KNN...")
        self.tree.fit(X, Y)
        #print("Training: SVD Tree...")
        self.knn.fit(X, Y)

        #print("Training: SVD Neural nets...")
        self.perceptron.fit(X, nn_Y, epochs=100, batch_size=10, verbose=0)
        self.mlp.fit(X, nn_Y, epochs=100, batch_size=10, verbose=0)

    def test(self, x):

        y_hat = {}
        for name, model in self.models.items():
            y_hat[name] = model.predict(x)

        return y_hat


if __name__ == '__main__':
    print('lmao')
