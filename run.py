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

from skimage.exposure import is_low_contrast

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

from core import generate_data
from train_test_split import get_images, train_test_split
from exemplar_model import create_model

# PARAMETERS for making this a FUNCTION later on

max_epochs = 2
batch_size = 256

# ----------------------------------------------

def generate_batches(h5, labels, batch_size):
    """
    generator for loading data from hdf5 file
    labels is list of tuples (label, index in hdf5)
    h5     is hdf5 file containing X_train in ["samples"]
    """
    i = 0  # batch-position in dataset
    while True:
        # get batch images (sorting b.c. hdf5 wants sorted indices)
        current = sorted(labels[i:i+batch_size], key=lambda x:x[1])
        ind     = [x[1] for x in current]
        X_train = h5['samples'][ind]
        Y_train = np.array([x[0] for x in current])
        yield (X_train, Y_train)
        i += batch_size
        if i >= len(labels):
            i = 0
            random.shuffle(labels)


if __name__ == "__main__":

    path="/home/cobalt/deepvis/project/toon/my_sets"
    h5path = os.path.join(path, "surrogate.hdf5")

    if os.path.isfile(h5path):
        h5 = h5py.File(h5path, 'r')
    else:
        if os.path.isfile(os.path.join(path, "good_distr.p")):
            train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
        else:
            imgs = get_images()
            train, test, val, score = train_test_split(imgs, 150, N=50000)

        h5 = generate_data(train, hdf5=h5path)


    #TODO: Put everything from here on in a function getting model, epochs, batch_size, hdf5)

    labels = h5['labels'][()] # read labels, discard emptys
    labels = labels[ labels[:,0] != 0 ]
    cat = np_utils.to_categorical(labels[:,0])
    nb_classes = cat.shape[1]
    print("nb_classes: {}".format(nb_classes))
    labels = list(zip(cat, list(labels[:,1])))

    # labels := list of all labels (and image indices for easy shuffeling)
    #   item[0] := categorical label vector
    #   item[1] := corresponding index of image

    network = create_model((3, 64, 64), nb_classes)
    network.compile(loss      = 'categorical_crossentropy',
                    optimizer = 'adadelta',
                    metrics   =['accuracy'])
    checkpointer = ModelCheckpoint(filepath=os.path.join(path, "weights-{epoch:02d}.hdf5"),
                                   verbose=1, save_best_only=True)
    #initializing
    epochs  = 0
    history = {
               'loss': [],
               'val_loss': [],
               'acc': [],
               'val_acc': []
              }
    random.shuffle(labels)

    #training
    while(epochs < 2):

        hist = network.fit_generator(generate_batches(h5, labels, batch_size),
                                     samples_per_epoch=len(labels),nb_epoch=1)
        epochs += 1
        print("Trained epoch #{}".format(epochs)) #TODO: colored output? :)
        #TODO: SVM validation
        # print("Trained one batch") #TODO: better verbosity



















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


