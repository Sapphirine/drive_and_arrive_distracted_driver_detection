# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:24:21 2016

@author: Yuhang Liu
"""

import os
import cv2
import glob
import pickle
import h5py
import math
import numpy as np
from keras.utils import np_utils



def get_im_cv2(path, img_rows, img_cols, color_type=1):
    
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)

    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_driver_data():

    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    f = open(path, 'r')
    while(1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):

    X_train = []
    y_train = []
    driver_id = []
    driver_data = get_driver_data()
    print('Read train images')

    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1):

    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
            
    return X_test, X_test_id


def cache_data(data, path):

    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def cache_train_data(train_data, train_target, driver_id, unique_drivers, path):

    if not os.path.isdir('cache'):
        os.mkdir('cache')
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("train_data", data=train_data)
        hf.create_dataset("train_target", data=train_target)
        hf.create_dataset("driver_id", data=driver_id)
        hf.create_dataset("unique_drivers", data=unique_drivers)


def cache_test_data(test_data, test_id, path):

    if not os.path.isdir('cache'):
        os.mkdir('cache')
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("test_data", data=test_data)
        hf.create_dataset("test_id", data=test_id)


def read_train_data(img_rows, img_cols, color_type=1):

    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.h5')
    
    train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
    train_data = np.array(train_data, dtype=np.uint8)
    
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    cache_train_data(train_data, train_target, driver_id, unique_drivers, cache_path)

    return train_data, train_target, driver_id, unique_drivers



def read_test_data(img_rows, img_cols, color_type=1):

    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.h5')

    test_data, test_id = load_test(img_rows, img_cols, color_type)
    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    cache_test_data(test_data, test_id, cache_path)

    return test_data, test_id



img_rows = 224
img_cols = 224
color_type = 3

read_train_data(img_rows, img_cols, color_type)

read_test_data(img_rows, img_cols, color_type)

#%%

img = cv2.imread("imgs/train/c2/img_381.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_381.jpg',img)

