#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:36:53 2017

@author: cobalt
"""

from keras.callbacks import Callback
from keras import backend as K

class SVM_Validation(Callback):

    def get_representation(self, model, layer, input_batch):
        """
        This is stolen from my Deep Vision exercises...
        I hope, that's okay...
        """
        get_representation = K.function([model.layers[0].input,K.learning_phase()],
                                        [model.layers[layer].output,])
        representation = get_representation([input_batch, 0])
        return representation

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #TODO: <insert coffee here>
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return