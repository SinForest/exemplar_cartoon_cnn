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
train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
test     = list(zip(*test))
val      = list(zip(*val))
train    = None
inp_size = 64
X_test   = np.array(list(map(dim_ordering_th, resize_all(test[0], inp_size))))
X_val    = np.array(list(map(dim_ordering_th, resize_all( val[0], inp_size))))
Y_test   = list(test[1])
Y_val    = list( val[1])
classes  = sorted(list(set(Y_test)))

batch_size = 128

i = 0
F_test = []
while i < len(X_test):
    F_test.extend(list(get_representation(network, X_test[i:i+batch_size])[0]))
    i += batch_size
i = 0
F_val = []
while i < len(X_val):
    F_val.extend(list(get_representation(network, X_val[i:i+batch_size])[0]))
    i += batch_size

svm = SVC()
svm.fit(F_test, Y_test)
acc = svm.score(F_val, Y_val)
print("Achieved SVM-Accuracy: {}".format(acc))

# PCA + t-SNE

# "It is highly recommended [...] to reduce the number of dimensions
#  to a reasonable amount (e.g. 50)"
# ~sklearn doc
pca = PCA(n_components=50)
vec = pca.fit_transform(F_test + F_val)
tsne = TSNE(method='exact', learning_rate=2000, n_iter=10000, perplexity=30)
vec = tsne.fit_transform(vec)
labels = np.array(Y_test + Y_val)

N      = len(classes)
S      = [0.95, 0.85, 0.75]
V      = [0.95, 0.75, 0.55]
hsv    = [(x/N, S[x%3], V[x%3]) for x in range(N)]
rgb    = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
colors = list(map(lambda x: "#{0:02x}{1:02x}{2:02x}".format(*map(lambda c:int(255*c),x)), rgb))

markers = ["*", "^", "p", "<", "o", ">", "h", "v", "D"]

fig, ax1 = plt.subplots()
fig.set_size_inches(7, 7)

for i in range(len(classes)):
    xy = vec[labels == classes[i]]
    x = xy[:,0]
    y = xy[:,1]
    ax1.plot(x, y, markers[i%len(markers)], c=colors[i], label=classes[i])

box = ax1.get_position()
ax1.set_position([box.x0, box.y0,
                  box.width * 0.8, box.height])
han, lab = ax1.get_legend_handles_labels()
fig.legend(han, lab, loc='center right',ncol=1)

plt.show()