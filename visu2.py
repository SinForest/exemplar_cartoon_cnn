#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 00:58:12 2017

@author: cobalt
"""

import os
import h5py
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
import random
import pickle
import colorsys
from core import dim_ordering_tf, resize_all

from keras.models import load_model
from keras.layers import Dense
from keras import backend as K

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from core import resize_all, dim_ordering_th

def get_representation(model, input_batch):
    """
    Extracts features from pre-last Dense layer (+PReLU)
    """
    #get index of activation layer after pre-last Dense layer
    layer = sorted([x for x in zip(range(len(model.layers)), model.layers)
                    if type(x[1]) is Dense])[-2][0] + 1

    get_representation = K.function([model.layers[0].input,K.learning_phase()],
                                    [model.layers[layer].output,])
    representation = get_representation([input_batch, 0])
    return representation


network = load_model('/home/cobalt/deepvis/project/toon/my_sets/best_weights.hdf5')
path="/home/cobalt/deepvis/project/toon/my_sets"
h5path = os.path.join(path, "surrogate.hdf5")
h5 = h5py.File(h5path, 'r')

num     = len(h5['samples'])
nums    = sorted(random.sample(range(num), 1024))
samples = h5['samples'][nums]

features = []
i = 0
batch_size = 128
while i < len(samples):
    features.extend(list(get_representation(network, samples[i:i+batch_size])[0]))
    i += batch_size

pca = PCA(n_components=50)
vec = pca.fit_transform(features)
tsne = TSNE(method='exact', learning_rate=2000, n_iter=10000, perplexity=30)
vec = tsne.fit_transform(vec)

vec -= vec.min()
vec /= vec.max()
samples = [dim_ordering_tf(sam) for sam in samples]
#samples = resize_all(samples, 8)

fig, ax = plt.subplots()

for i in range(len(vec)):
    bb = Bbox.from_bounds(vec[i][0], vec[i][1], 0.05, 0.05)
    bb2 = TransformedBbox(bb, ax.transData)
    bbox_image = BboxImage(bb2, norm = None, origin=None, clip_on=False)
    bbox_image.set_data(samples[i])
    ax.add_artist(bbox_image)

fig.set_size_inches(20, 10)
plt.imsave("plot_images.png")
