#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:44:02 2017

@author: cobalt
"""

import os
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize, rotate

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

DEBUG = "data"

def create_model(input_shape, nb_classes):
    model = Sequential()
    
    model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=input_shape))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(128, 5, 5))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(256, 5, 5))
    model.add(PReLU())
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model
    
def dim_ordering_th(A):
    if A.shape[0] in [1,3]:
        return np.swapaxes(np.swapaxes(A, 1, 3), 2, 3)
    else:
        raise ValueError("needs to be in tf-ordering")
    
def generate_data(path, max_side=128, crop_size=64, nb_samples=200, outpath=None):
    
    for filename in os.listdir(path):
        
        # read Image, if it's a file
        file = os.path.join(path, filename)
        if not os.path.isfile(file):
            continue
        img  = imread(file)
        
        # resizing
        if img.shape[0] > img.shape[1]:
            img = resize(img, (max_side, int(img.shape[1] * max_side / img.shape[0])))
        else:
            img = resize(img, (int(img.shape[0] * max_side / img.shape[1]), max_side))
        if img.shape[0] < crop_size or img.shape[1] < crop_size:
            raise RuntimeError("image '{}' is resized smaller than crop_size")
            
        # generate samples
        for i in range(nb_samples):
            
            # rotate image
            rot = np.random.randint(-20, 20)
            sample = rotate(img, rot, resize=False)
            
            # cropping random patch
            crop_y = np.random.randint(img.shape[0] - crop_size + 1)
            crop_x = np.random.randint(img.shape[1] - crop_size + 1)
            sample = sample[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]

            # flipping (maybe)
            if np.random.choice([True, False]):
                sample = np.fliplr(sample)
            
            

            
        

if __name__ == "__main__":
    
    if DEBUG == "model":
        nb_c = 585
        w_h = 32
        
        net = create_model((3, w_h, w_h), nb_c)
        net.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
        net.summary()
        
    if DEBUG == "data":
        generate_data("/home/cobalt/deepvis/project/toon/JPEGImages")

def evaluate_data(path):
    res = {"max_height": 0,
           "max_width": 0,
           "max_area": 0,
           "max_box": (0,0),
           "max_image": "",
           "sum_height": [],
           "sum_width": [],
           "sum_area": [],
          }
    if path[-1] != "/":
        path.append("/")
         
    for filename in os.listdir(path):
        if filename[0] == ".":
            continue
        img = imread(path + filename)
        res["sum_height"].append(img.shape[0])
        if img.shape[0] > res["max_height"]:
            res["max_height"] = img.shape[0]
        res["sum_width"].append(img.shape[1])
        if img.shape[1] > res["max_width"]:
            res["max_width"] = img.shape[1]
        res["sum_area"].append(img.shape[0]*img.shape[1])
        if img.shape[0]*img.shape[1] > res["max_area"]:
            res["max_area"] = img.shape[0]*img.shape[1]
            res["max_box"] = (img.shape[0], img.shape[1])
            res["max_image"] = filename
    
    for key in res:
        if type(res[key]) is list:
            res[key].sort()
            print("{}(SUM): {}".format(key, sum(res[key]) / len(res[key])))
        print("{}: {}".format(key, res[key]))
        