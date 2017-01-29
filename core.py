#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:44:02 2017

@author: cobalt
"""

import os
import h5py
import time
import random
import numpy as np

from skimage.io import imshow
from skimage.transform import resize, rotate
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import is_low_contrast

from keras import backend as K


DEBUG = "data"
#np.random.seed(1338) # for debugging

def generate_batches(h5, labels, batch_size):
    """
    GENERATOR for loading data from hdf5 file
    labels is list of tuples (label, index in hdf5)
    h5     is hdf5 file containing X_train in ["samples"]
    """
    i = 0  # batch-position in dataset
    random.shuffle(labels)
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

def dim_ordering_th(A):
    if A.shape[2] in [1,3]:
        return np.swapaxes(np.swapaxes(A, 0, 2), 1, 2)
    else:
        raise ValueError("needs to be in tf-ordering")

def dim_ordering_tf(A):
    if A.shape[0] in [1,3]:
        return np.swapaxes(np.swapaxes(A, 0, 1), 1, 2)
    else:
        raise ValueError("needs to be in th-ordering")

def train_exemplar_cnn(train, test, val):
    #train = zip()
    pass

def resize_all(X, size):
    res = []
    for img in X:
        res.append(resize(img, (size, size)))
    return np.array(res)


def generate_data(data, max_side=192, crop_size=64, nb_samples=200, zoom_lower = 0.7,
                  zoom_upper = 1.4, rotation = 20, stretch = 0.15, contrast = (1.5, 1.4, 0.1),
                  color = 0.05, hdf5=None):

    # old approach:
    """
    for filename in os.listdir(path):
        if filename[0] == ".":
            continue
        # read Image, if it's a file
        file = os.path.join(path, filename)
        if not os.path.isfile(file):
            continue
        img  = imread(file)
    """
    if hdf5 == None:
        pairs    = []
    else:
        h5file   = h5py.File(hdf5, 'x') # this will fail if file exists
        h5pairs  = h5file.create_dataset("samples", (nb_samples * len(data), 3, crop_size, crop_size))
        h5labels = h5file.create_dataset("labels", (nb_samples * len(data), 2)) # labels getting indices for easier shuffeling

    surrogate_class = 1 # begin with 1, so everything with class 0 is empty data
    i_sample = 0
    clock = time.clock()
    ete = float("inf")

    for img in data:

        print("### Generating Surrogate Class {} of {}  [ETE:{:.1f}m]###".format(surrogate_class, len(data), ete))

        # calculate resize factor (= factor to scale image, so largest side has size 128)
        re_fac = max_side / max(img.shape[0], img.shape[1])
        if crop_size / re_fac > min(img.shape[0], img.shape[1]):
            raise RuntimeError("can't crop on image\nSize: {}, re_fac:{}, crop_size:{}"
                               .format(img.shape, re_fac, crop_size))

        # generate samples
        for i in range(nb_samples):

            # rotate image
            rot = np.random.randint(-rotation, rotation)
            sample = rotate(img, rot, resize=False)

            # calculate zoom level and stretching
            # (zoom    > 1  -> zoom in,  zoom    < 1  -> zoom out)
            # (stretch < 0  -> taller,   stretch > 0  -> wider)
            while True:
                zoom = np.random.ranf() * (zoom_upper - zoom_lower) + zoom_lower
                stretch = np.random.ranf() * stretch * 2 - stretch
                zoom_crop_x = int(crop_size / (zoom * re_fac * (1+stretch)))
                zoom_crop_y = int(crop_size / (zoom * re_fac * (1-stretch)))
                if zoom_crop_x > img.shape[1] or zoom_crop_y > img.shape[0]:
                    continue
                break

            # cropping random patch
            crop_y = np.random.randint(img.shape[0] - zoom_crop_y + 1)
            crop_x = np.random.randint(img.shape[1] - zoom_crop_x + 1)
            sample = sample[crop_y:crop_y+zoom_crop_y, crop_x:crop_x+zoom_crop_x, :]

            # resize patch to crop_size x crop_size
            sample = resize(sample, (crop_size, crop_size))

            # flipping (maybe)
            if np.random.choice([True, False]):
                sample = np.fliplr(sample)

            # alter contrast
            sample = rgb2hsv(sample)
            pot = np.random.ranf() * (contrast[0] - 1) + 1
            if np.random.choice([True, False]):
                pot = 1 / pot
            mul = np.random.ranf() * (contrast[1] - 1/contrast[1]) + 1/contrast[1]
            add = np.random.ranf() * contrast[2] * 2 - contrast[2]

            sample[:,:,1:] = sample[:,:,1:] ** pot
            sample[:,:,1:] = sample[:,:,1:] * mul
            sample[:,:,1:] = sample[:,:,1:] + add
            sample[sample > 1] = 1
            sample[sample < 0] = 0

            #alter hue
            hue = np.random.ranf() * color * 2 - color
            sample[:,:,0] = sample[:,:,0] + hue
            sample[sample > 1] = sample[sample > 1] - 1
            sample[sample < 0] = sample[sample < 0] + 1

            # convert back to RGB
            sample = hsv2rgb(sample)

            # discard images with low contrast
            if is_low_contrast(sample):
                continue

            # change dim ordering if needed
            if K.image_dim_ordering() == 'th':
                sample = dim_ordering_th(sample)

            # save depending on save-method
            if hdf5 == None:
                pairs.append((sample, surrogate_class))
            else:
                h5pairs[i_sample,:,:,:] = sample
                h5labels[i_sample,0] = surrogate_class
                h5labels[i_sample,1] = i_sample # labels getting indices for easier shuffeling (²)
            i_sample += 1


            # debug
            continue
            imshow(sample)
            return

        # end for i in range(nb_samples)

        ete = ((time.clock() - clock) / surrogate_class) * (len(data) - surrogate_class) / 60
        surrogate_class += 1

    # end for img in data


    if hdf5 == None:
        return pairs
    else:
        #h5pairs.resize(i_sample)
        #h5labels.resize(i_sample)
        return h5file


    # plotting for debug
    """
    plt.figure(figsize=(20, 20), dpi=80)
    for i in range(nb_samples):
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(samples[i])
    plt.savefig("batches.png")
    return
    """




if __name__ == "__main__":

    if DEBUG == "data":
        generate_data("/home/cobalt/deepvis/project/toon/JPEGImages")