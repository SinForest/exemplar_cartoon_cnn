#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:39:48 2017

@author: cobalt
"""
import os
import pickle

from skimage.exposure import is_low_contrast

from core import generate_data
from train_test_split import get_images, train_test_split

if __name__ == "__main__":
    
    path="/home/cobalt/deepvis/project/toon/my_sets"
    
    if os.path.isfile(os.path.join(path, "good_distr.p")):
        train, test, val = pickle.load(open(os.path.join(path, "good_distr.p"), "rb"))
    else:
        imgs = get_images()
        train, test, val, score = train_test_split(imgs, 150, N=50000)
    
    pairs = generate_data(train)

        
    
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
    
    print("### Pickling Surrogate Classes ###")
    pickle.dump(pairs, open(os.path.join(path, "surrogate.p"), "wb"))
    print("### Finished Pickling ###")
    