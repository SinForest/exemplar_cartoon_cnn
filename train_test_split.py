#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:39:10 2017

@author: cobalt
"""

import os, re
import random

from skimage.io import imread, imshow


def get_images(path="/home/cobalt/deepvis/project/toon/Annotations"):
    """
    returns list of dicts
        each dict has
        ["img"] with filename
        ["boxes"] as list of dicts
            each dict has
            ["name"] with class
            ["xmin"]["xmax"]["ymin"]["ymax"] bounding box
    """
    res = []

    for filename in os.listdir(path):

        if filename[0] == ".":
            continue
        # read Image, if it's a file
        file = os.path.join(path, filename)
        if not os.path.isfile(file):
            continue
        file  = open(file)
        xml   = file.read()
        match = re.search("<filename>(.*)<\/filename>", xml)
        if not match:
            print("Error with file {}: no match for filename".format(filename))
            continue
        d = {}
        d["img"] = match.group(1)
        
        try:
            it = re.finditer("<object>[\w\W]*?<\/object>", xml)
            d["boxes"] = []
            for match in it:
                tmp = {}
                tmp["name"] = re.search("<name>(.*)<\/name>", match.group(0)).group(1)
                tmp["xmin"] = int(float(re.search("<xmin>(.*)<\/xmin>", match.group(0)).group(1)))
                tmp["xmax"] = int(float(re.search("<xmax>(.*)<\/xmax>", match.group(0)).group(1)))
                tmp["ymin"] = int(float(re.search("<ymin>(.*)<\/ymin>", match.group(0)).group(1)))
                tmp["ymax"] = int(float(re.search("<ymax>(.*)<\/ymax>", match.group(0)).group(1)))
                d["boxes"].append(tmp)
        except:
            print("Error at file {}: attributes not found".format(filename))
        
        res.append(d)
            
    return res
                

def train_test_split(data, test_count, path="/home/cobalt/deepvis/project/toon/JPEGImages"):
    """
    returns (train, test) with
        train: list of filenames for training
        test : list of bounding-box crops as np-Arrays
    """
    count = test_count
    train = data.deepcopy()
    random.shuffle(train)
    
    for i in range(train):
        if count > 0:
            break
        for box in train[i]["boxes"]:
            #img = imread()
            pass #TODO

get_images()