#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:33:44 2017

@author: cobalt
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import colorsys

from keras.models import load_model
from keras.layers import Dense
from keras import backend as K

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
test   = list(zip(*test))
train = None
val = None
inp_size = 64
X_test = np.array(list(map(dim_ordering_th, resize_all(test[0], inp_size))))
Y_test = np.array(test[1])
classes = sorted(list(set(Y_test)))

i = 0
batch_size = 128
F_test = []
while i < len(X_test):
    F_test.extend(list(get_representation(network, X_test[i:i+batch_size])[0]))
    i += batch_size

# PCA + t-SNE

# "It is highly recommended [...] to reduce the number of dimensions
#  to a reasonable amount (e.g. 50)"
# ~sklearn doc
pca = PCA(n_components=50)
F_test = pca.fit_transform(F_test)
tsne = TSNE()
F_test = tsne.fit_transform(F_test)

N      = len(classes)
hsv    = [(x/N, 0.95, 0.95) for x in range(N)]
rgb    = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
colors = list(map(lambda x: "#{0:02x}{1:02x}{2:02x}".format(*map(lambda c:int(255*c),x)), rgb))

for i in range(len(classes)):
    xy = F_test[Y_test == classes[i]]
    x = xy[:,0]
    y = xy[:,1]
    plt.plot(x, y, "o", c=colors[i])

plt.show()