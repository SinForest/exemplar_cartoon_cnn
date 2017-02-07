#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:36:53 2017

@author: cobalt
"""

from keras.layers import Dense
from keras.callbacks import Callback
from keras import backend as K
from sklearn.svm import SVC

class SVM_Validation(Callback):

    def __init__(self, X_val, Y_val, X_test, Y_test, batch_size):
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.batch_size = batch_size

    def get_representation(self, input_batch):
        """
        Extracts features from pre-last Dense layer (+PReLU)
        """
        #get index of activation layer after pre-last Dense layer
        layer = sorted([x for x in zip(range(len(self.model.layers)), self.model.layers)
                        if type(x[1]) is Dense])[-2][0] + 1

        get_representation = K.function([self.model.layers[0].input,K.learning_phase()],
                                        [self.model.layers[layer].output,])
        representation = get_representation([input_batch, 0])
        return representation

    def on_train_begin(self, logs={}):
        self.my_log = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        i = 0
        F_val = []
        while i < len(self.X_val):
            F_val.extend(list(self.get_representation(self.X_val[i:i+self.batch_size])[0]))
            i += self.batch_size
        i = 0
        F_test = []
        while i < len(self.X_test):
            F_test.extend(list(self.get_representation(self.X_test[i:i+self.batch_size])[0]))
            i += self.batch_size
        svm = SVC()
        svm.fit(F_val, self.Y_val)
        acc = svm.score(F_test, self.Y_test)
        print("Achieved SVM-Accuracy: {}".format(acc))
        self.my_log.append(acc)
        #TODO: add to models val_loss (and turn save_best_only on again)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return