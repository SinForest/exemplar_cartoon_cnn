#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:44:02 2017

@author: cobalt
"""

import os
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize, rotate
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import is_low_contrast


DEBUG = "data"
#np.random.seed(1338) # for debugging
    
def dim_ordering_th(A):
    if A.shape[0] in [1,3]:
        return np.swapaxes(np.swapaxes(A, 1, 3), 2, 3)
    else:
        raise ValueError("needs to be in tf-ordering")

def train_exemplar_cnn(train, test, val):
    #train = zip()
    pass
    
def generate_data(data, max_side=192, crop_size=64, nb_samples=200, zoom_lower = 0.7,
                  zoom_upper = 1.4, rotation = 20, stretch = 0.15, contrast = (1.5, 1.4, 0.1),
                  color = 0.05, outpath=None):
    
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
    
    pairs = []
    surrogate_class = 0
    
    for img in data:
        
        print("### Generating Surrogate Class {} of {} ###".format(surrogate_class + 1, len(data)))
        
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
            
            #discard images with low contrast
            if not is_low_contrast(sample):
                pairs.append((sample, surrogate_class))
            
            # debug
            continue
            imshow(sample)
            return
        
        # end for i in range(nb_samples) 
        
        surrogate_class += 1
    
    # end for img in data
    
    return pairs
        
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