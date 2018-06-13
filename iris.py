import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import np_utils

def main():

  iris = datasets.load_iris()
  features = iris.data
  targets = iris.target

  x = preprocessing.scale(features)
  y = np_utils.to_categorical(targets)

  model = Sequential()
  model.add(Dense(12, input_dim=4))
  model.add(Activation('sigmoid'))
  model.add(Dense(3, input_dim=12))
  model.add(Activation('softmax'))
  model.compile(optimizer='SGD', loss='binary_crossentropy', \
                metrics=['accuracy'])

  model.fit(x, y, nb_epoch=5000, batch_size=5, verbose=1)
  model.predict(x, batch_size=1, verbose=1)

  model.metrics_names
  score = model.evaluate(x, y, batch_size=1)

  print("loss = ", score[0])
  print("accuracy = ", score[1])

if __name__ == "__main__":

  main()
