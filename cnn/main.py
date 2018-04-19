#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping

def layers():

  print("read layer")
  model = Sequential()

  model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(256, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(256, (3, 3)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))

  model.add(Conv2D(512, (3, 3)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(1))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

  return model

def main():

  '''
  """""""""""""""""""""""""""""
  X_train : learning data
  Y_train : learning label
  X_test  : test data
  Y_test  : test label

  """""""""""""""""""""""""""""
  '''

  train_dataset = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, \
                                      zoom_range=0.2)


  test_dataset = ImageDataGenerator(rescale=1.0 / 255)

  # 2class predict : class_mode='binary'
  # nclass predict : calss_mode='categorical'
  train_generator = train_dataset.flow_from_directory('dataset/train', target_size=(224, 224), \
                                                        batch_size=1, class_mode='binary')

  test_generator = test_dataset.flow_from_directory('dataset/validation', target_size=(150, 150), \
                                                        batch_size=1, class_mode='binary')

  print("!!!!!!!!!!!!!!")
  # define CNN layers
  model = layers()
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #early = EarlyStopping()

  '''
  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  fit() : When train by prepared Numpy array's datasets
  fit_generator : When train by datasets that original image

  training 
  samples_per_epoch : batch size
  nb_val_samples : bacth size

  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  '''

  history = model.fit_generator(train_generator, steps_per_epoch=2000, \
                            epochs=10, validation_data=test_generator, validation_steps=800)

  #history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
  #                    verbose=1, validation_data=(x_test, y_test), callbacks=[early])

  score = model.evaluate_generator(generator=test_generator)
  print('test loss:', score[0])
  print('test acc:', score[1])

  # save weights
  model.save_weights(os.path.join(result_dir, 'model.h5'))
  #save_history(history, os.oath.join(result_dir, 'history_vgg.txt'))  
  
if __name__ == '__main__':
  
  main()  
  

