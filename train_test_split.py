#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:39:10 2017

@author: cobalt
"""

import os, re
import random
import itertools
import pickle

from skimage.io import imread, imsave


def get_images(path="/home/cobalt/deepvis/project/toon/Annotations"):
    """
    returns list of dicts
        each dict has
        ["file"] with filename
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
        d["file"] = match.group(1)
        
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
    
def extract_classes(data, asSet=False):
    res = [box["name"] for box in 
           itertools.chain.from_iterable(
           [image["boxes"] for image in data])
          ]
    return set(res) if asSet else res
                

def extract_training_pairs(data, count=None):
    pairs    = []
    random.shuffle(data)
    
    for i in range(len(data)):
        if count != None and count < 0:
            break # when enough test samples are made

        img =  data[i]["file"]
        # crop bboxes
        for box in data[i]["boxes"]:
            x0 = box["xmin"]
            y0 = box["ymin"]
            x1 = box["xmax"]
            y1 = box["ymax"] 
            crop = img[y0:y1, x0:x1, :]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            pairs.append((crop, box["name"])) # X/y pairs
            if count != None:
                count -= 1
        data[i] = None

    # delete entries of test images
    data = [x for x in data if x != None]
    if count != None:
        return pairs, data
    else:
        return pairs
    
    

def train_test_split(data, test_count, N=1000, path="/home/cobalt/deepvis/project/toon/JPEGImages"):
    """
    returns (train, test) with
        train: list of filenames for training
        test : list of bounding-box crops as np-Arrays
    """
    
    # initialize
    best_score        = 2
    best_distribution = (None, None)
    skip_counter = 0
    better_counter = 0
    
    # load images, clean data from 404s
    for i in range(len(data)):
        try:
            data[i]["file"] = imread(os.path.join(path, data[i]["file"]))
        except FileNotFoundError:
                data[i] = None
    data = [x for x in data if x != None]    

    # make N iterations: brute-force-approach
    for step in range(N): 
        if step % 100 == 0 and __name__ == "__main__":
            print("Iter #{}".format(step))
        
        # initialize iteration
        classes = extract_classes(data, asSet=True)
        train   = data[:]
        
        test, train = extract_training_pairs(train, test_count)
        
        #if train == best_distribution[0] or test == best_distribution[1]:
            #print("Equality in iter {}".format(step))
        # calculate scores
        score = 0
        train_classes = extract_classes(train)
        test_classes = [Xy[1] for Xy in test]
        try:
            for cl in classes:
                train_score  = train_classes.count(cl) / len(train_classes) - 1 / len(classes)
                test_score   = test_classes.count(cl) / len(test_classes) - 1 / len(classes)
                score += test_score ** 2 + train_score ** 2
        except ZeroDivisionError:
            score = 2
            skip_counter += 1
            print([cl for cl in classes if cl in test_classes])
                
        if score < best_score:
            best_score = score
            best_distribution = (train, test)
            better_counter += 1
    
    #create validation set (like testset with training images)
    val = train[:]
    val = extract_training_pairs(val)
    
    if __name__ == "__main__": print("Skipped:", skip_counter)
    if __name__ == "__main__": print("Better:", better_counter)
    return [item["file"] for item in best_distribution[0]], best_distribution[1], val, score
        
def extract_pickle_set(filename, path="/home/cobalt/deepvis/project/toon/my_sets"):
    file = os.path.join(path, filename)
    for fol in ["train", "test", "val"]:
        tmp = os.path.join(path, fol)
        if not os.path.isdir(tmp):
            os.makedirs(tmp)
            
    train, test, val = pickle.load(open(file, "rb"))
    print("Train:", len(train), "Test:", len(test), "Val:", len(val))
    
    i = 0
    for img in train:
        file = os.path.join(path, "train", "train_{:03d}.bmp".format(i))
        imsave(file, img)
        i += 1
        
    i = 0
    txt = open(os.path.join(path, "test.txt"), "w")
    for Xy in test:
        file = os.path.join(path, "test", "test_{:03d}.bmp".format(i))
        imsave(file, Xy[0])
        txt.write("test_{:03d}.bmp\t{}\n".format(i, Xy[1]))
        i += 1
    txt.close()
    
    i = 0
    txt = open(os.path.join(path, "val.txt"), "w")
    for Xy in val:
        file = os.path.join(path, "val", "val_{:03d}.bmp".format(i))
        imsave(file, Xy[0])
        txt.write("val_{:03d}.bmp\t{}\n".format(i, Xy[1]))
        i += 1
    txt.close()
        
    
if __name__ == "__main__":
    imgs = get_images()
    train, test, val, score = train_test_split(imgs, 150, N=50000)
    test_classes = [x[1] for x in test]
    print("Score:", score)
    print("Classes:", len(test_classes))
    print("Set:", len(set(test_classes)))
    for cl in set(test_classes):
        print(cl, "\t:\t", test_classes.count(cl))
        
    print("Train:", len(train), "Test:", len(test), "Val:", len(val))
    pickle.dump((train, test, val), open("/home/cobalt/deepvis/project/toon/best_distr_{:.2f}.p".format(score), "wb"))