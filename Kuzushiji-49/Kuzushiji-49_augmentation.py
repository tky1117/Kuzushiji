import os
import zipfile
import numpy as np
import pandas as pd
import glob
import keras
from keras.preprocessing.image import load_img, ImageDataGenerator

n_class = 49

if __name__ == '__main__':
    train_X_file = np.load(os.path.join('data', 'k49-train-imgs.npz'))
    train_X = train_X_file['arr_0']
    train_X = np.expand_dims(train_X, axis = 3)

    if not os.path.exists(os.path.join('data', 'k49-train_rotate-imgs.npz')):
        image_data_generator = ImageDataGenerator(fill_mode = 'nearest', rotation_range = 10.0)
        generator = image_data_generator.flow(train_X, shuffle = False, batch_size = len(train_X))
        train_rotate_X = generator.next()
        train_rotate_X = np.reshape(train_rotate_X, train_rotate_X.shape[: 3])
        np.savez_compressed(save_path, arr_0 = train_rotate_X.astype(np.uint8))
