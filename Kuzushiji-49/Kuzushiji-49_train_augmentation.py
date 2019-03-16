'''
    pwd
    |- Kuzushiji-49_train.py
    |- data
        |- k49-train-imgs.npz
        |- k49-train-labels.npz
'''

import os
import zipfile
import numpy as np
import pandas as pd
import glob
import keras
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger

n_class = 49
img_shape = (28, 28, 1)

def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(inputs)
    x = BatchNormalization()(x)
    outputs = Activation('relu')(x)
    
    return outputs

class CNN:
    def __init__(self, model = None, compiled = True):
        self.img_shape = img_shape
        
        optimizer = keras.optimizers.Adam()
        
        if model is None:
            self.model = base_CNN_model()
            self.model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        else:
            self.model = model
            if not compiled:
                self.model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

        self.model.summary()
    
    def train(self, epochs, save_path, model_name):
        
        train_X_file = np.load(os.path.join(main_path, 'k49-train-imgs.npz'))
        train_rotate_X_file = np.load(os.path.join(main_path, 'k49-train_rotate-imgs.npz'))
        train_Y_file = np.load(os.path.join(main_path, 'k49-train-labels.npz'))
        
        train_X = np.expand_dims(train_X_file['arr_0'], axis = 3) / 255.0
        train_rotate_X = np.expand_dims(train_rotate_X_file['arr_0'], axis = 3) / 255.0
        train_Y = np_utils.to_categorical(train_Y_file['arr_0'], n_class)
        
        train_X = np.concatenate([train_X, train_rotate_X], axis = 0)
        train_Y = np.concatenate([train_Y, train_Y], axis = 0)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        if not os.path.exists(os.path.join(save_path, model_name)):
            os.mkdir(os.path.join(save_path, model_name))
        
        checkpoint = ModelCheckpoint(os.path.join(save_path, model_name, model_name + '_{epoch:d}.h5'))
        csv_logger = CSVLogger(os.path.join(save_path, model_name + '.csv'), append = True)
        self.model.fit(train_X, train_Y, epochs = epochs, callbacks = [checkpoint, csv_logger], validation_split = 0.1)

def base_CNN_model():
    #define CNN structure
    inputs = Input(shape = img_shape)
    
    x = cba(inputs, filters = 64, kernel_size = (2, 2), strides = (1, 1))
    x = cba(x, filters = 64, kernel_size = (3, 3), strides = (2, 2))
    x = cba(x, filters = 128, kernel_size = (3, 3), strides = (2, 2))
    x = cba(x, filters = 256, kernel_size = (3, 3), strides = (2, 2))
        
    x = GlobalAveragePooling2D()(x)
            
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
            
    x = Dense(n_class)(x)
            
    outputs = Activation('softmax')(x)
            
    return Model(inputs, outputs)

if __name__ == '__main__':
    cnn = CNN()
    cnn.train(epochs = 100, save_path = 'model', model_name = 'CNN_model_1_augmentation')
