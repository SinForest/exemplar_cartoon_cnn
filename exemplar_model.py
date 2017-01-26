#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:25:43 2017

@author: cobalt
"""

DEBUG = "model"

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

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
    
if __name__ == "__main__":
    
    if DEBUG == "model":
        nb_c = 585
        w_h = 64
        
        net = create_model((3, w_h, w_h), nb_c)
        net.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
        net.summary()