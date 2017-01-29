#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:39:48 2017

@author: cobalt
"""
import os
import h5py
import pickle
import numpy as np
import random

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

from core import generate_data, resize_all, dim_ordering_th, generate_batches
from train_test_split import get_images, train_test_split
from exemplar_model import create_model
from svm_val import SVM_Validation

# PARAMETERS for making this a FUNCTION later on

max_epochs = 5
batch_size = 256
inp_size = 64

# ----------------------------------------------

def string_categorical(val, test):
    """
    returns test and val set Y as categorical
    only nedded if labels are strings!
    #TODO: implement directly into data generation! who needs string labels anyways?
    """
    classes = sorted(list(set(val + test)))
    val  = [classes.index(x) for x in val]
    test = [classes.index(x) for x in test]
    return val, test

if __name__ == "__main__":

    path="/home/cobalt/deepvis/project/toon/my_sets"
    h5path = os.path.join(path, "surrogate.hdf5")

    if os.path.isfile(os.path.join(path, "good_distr.p")):
        train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
    else:
        imgs = get_images()
        train, test, val, score = train_test_split(imgs, 150, N=50000)

    if os.path.isfile(h5path):
        h5 = h5py.File(h5path, 'r')
    else:
        h5 = generate_data(train, hdf5=h5path)

    #TODO: Put everything from here on in a function getting model, epochs, batch_size, hdf5)

    test   = list(zip(*test))
    val    = list(zip(*val))

    X_val  = np.array(list(map(dim_ordering_th, resize_all( val[0], inp_size))))
    X_test = np.array(list(map(dim_ordering_th, resize_all(test[0], inp_size))))
    Y_val, Y_test  = string_categorical(val[1], test[1])

    labels = h5['labels'][()] # read labels, discard emptys
    labels = labels[ labels[:,0] != 0 ]
    cat = np_utils.to_categorical(labels[:,0])
    nb_classes = cat.shape[1]
    labels = list(zip(cat, list(labels[:,1])))

    # labels := list of all labels (and image indices for easy shuffeling)
    #   item[0] := categorical label vector
    #   item[1] := corresponding index of image

    network = create_model((3, inp_size, inp_size), nb_classes)
    network.compile(loss      = 'categorical_crossentropy',
                    optimizer = 'adadelta',
                    metrics   =['accuracy'])
    mcp  = ModelCheckpoint(filepath=os.path.join(path, "weights-{epoch:02d}.hdf5"),
                          verbose=1, save_best_only=False)
    svmv = SVM_Validation(X_val, Y_val, X_test, Y_test, 128)
    #initializing
    history = {
               'loss': [],
               'val_loss': [],
               'acc': [],
               'val_acc': []
              }
    random.shuffle(labels)

    #training

    hist = network.fit_generator(generate_batches(h5, labels, batch_size),
                                 samples_per_epoch=len(labels),nb_epoch=max_epochs,
                                 callbacks=[mcp, svmv])

    """
    i = 0
    txt = open(os.path.join(path, "surrogate.txt"), "w")
    for Xy in pairs:
        file = os.path.join(path, "surrogate", "surrogate_{:03d}.bmp".format(i))
        imsave(file, Xy[0])
        txt.write("surrogate_{:03d}.bmp\t{}\n".format(i, Xy[1]))
        i += 1
    txt.close()
    """