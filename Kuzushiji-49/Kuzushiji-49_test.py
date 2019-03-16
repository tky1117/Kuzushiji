'''
    pwd
    |- Kuzushiji-49_train.py
    |- data
        |- k49-test-imgs.npz
        |- k49-test-labels.npz
    |- model
        |- CNN_model_1_1.h5
        |- CNN_model_1_2.h5
        ...
        |- CNN_model_1_100.h5
'''

import os
import zipfile
import numpy as np
import pandas as pd
import glob
import keras
from keras.utils import np_utils
from keras.models import load_model

n_class = 49

if __name__ == '__main__':
    save_path = 'model'
    CNN_name = 'CNN_model_1'
    model_name = 'CNN_model_1_100.h5'
    model = load_model(os.path.join(save_path, CNN_name, model_name))
    
    test_X_file = np.load(os.path.join('data', 'k49-test-imgs.npz'))
    test_Y_file = np.load(os.path.join('data', 'k49-test-labels.npz'))
    test_X = np.expand_dims(test_X_file['arr_0'], axis = 3) / 255.0
    test_Y = np_utils.to_categorical(test_Y_file['arr_0'], n_class)
    
    loss, acc = model.evaluate(test_X, test_Y)

    print('test')
    print('loss: {:f}, acc.: {:f}'.format(loss, acc))

