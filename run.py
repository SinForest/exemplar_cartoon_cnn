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
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

from core import generate_data, resize_all, dim_ordering_th, generate_batches
from train_test_split import get_images, train_test_split
from exemplar_model import create_model, create_model_deeper, create_model_bn
from svm_val import SVM_Validation

# PARAMETERS for making this a FUNCTION later on

max_epochs = 30
batch_size = 128
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

    print("### Beginning ###")
    path="/home/cobalt/deepvis/project/toon/my_sets"
    h5path = os.path.join(path, "surrogate.hdf5")

    if os.path.isfile(os.path.join(path, "good_distr.p")):
        print("### Unpickeling ###")
        train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
    else:
        print("### Create Test-Train-Set ###")
        imgs = get_images()
        train, test, val, score = train_test_split(imgs, 150, N=50000)

    if os.path.isfile(h5path):
        print("### Reading hdf5-File ###")
        h5 = h5py.File(h5path, 'r')
    else:
        print("### Generate Data ###")
        h5 = generate_data(train, hdf5=h5path)

    #TODO: Put everything from here on in a function getting model, epochs, batch_size, hdf5)
    print("### Doing Python Stuff ###")
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

    print("### Generate Network ###")
    network = create_model_bn((3, inp_size, inp_size), nb_classes)
    print("### Compile Network ###")
    network.compile(loss      = 'categorical_crossentropy',
                    optimizer = 'adadelta',
                    metrics   =['accuracy'])
    mcp  = ModelCheckpoint(filepath=os.path.join(path, "weights-{epoch:02d}.hdf5"),
                          verbose=1, save_best_only=False)
    svmv = SVM_Validation(X_val, Y_val, X_test, Y_test, 128)

    #training

    print("### Training... ###")
    hist = network.fit_generator(generate_batches(h5, labels, batch_size),
                                 samples_per_epoch=len(labels),nb_epoch=max_epochs,
                                 callbacks=[mcp, svmv])
    print(svmv.my_log)



    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 5)
    plt.title('Accuracy & Loss')
    ax1.set_xlabel("Epochs")
    # summarize history for accuracy
    lns = ax1.plot(hist.history['acc'], c='m')
    lns += ax1.plot(svmv.my_log, c='r')
    ax1.set_ylabel('accuracy', color='r')
    ax1.set_ylim(0, 1)

    # summarize history for loss
    ax2 = ax1.twinx()
    lns += ax2.plot(hist.history['loss'], c='c')
    ax2.set_ylabel('loss', color='b')

    # make it beautiful *-*
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    fig.legend(lns, ['acc train', 'acc svm-val', 'loss train'],
               loc='upper center', bbox_to_anchor=(0.5, 0.10),ncol=4,
               fancybox=True, shadow=True)
    plt.savefig("tmp_plot_training_curve.svg")