import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

from PIL import Image

import glo_var


# This class defines how to read from disk and preprocess the image
class ImageGen(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        # for each item in the outter list, they each coordinates to respective output FC layer with *softmax* as its activation
        batch_ys = list(np.transpose(self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]))
        
        image_arrs = []
        for filename in batch_x:
            image_arrs.append(Image2Tensor(glo_var.image_dir + str(filename)))
        
        return (np.array(image_arrs), batch_ys)


# `ABCD` -> `[0, 1, 2, 3]`
def Chars2Index(chars):
    index = []
    count = 0

    for char in chars:
        for word in glo_var.chars:
            if word == char:
                index.append(count)
            count += 1
        count = 0

    return index


# `[0, 1, 2, 3]` -> `ABCD`
def Index2Chars(label):
    chars = []
    for index in label:
        chars.append(glo_var.chars[index])

    string = "".join(chars)

    return string


# PIL form -> np array form
def Image2Tensor(imagePath):
    image_arr = np.array(Image.open(imagePath))
    image_arr = image_arr.astype(np.float32)
    image_arr = image_arr - 147.0
    # alternative tunning
    # image_arr = image_arr / 255.0
    # image_arr = image_arr - 0.5

    return image_arr
    

# Scan dir and get all images and labels for train
def get_filenames_labels():
    labels = []
    for _, _, filenames in os.walk(glo_var.image_dir):
        for filename in filenames:
            label_literal = filename.split('.')[0].split('_')[-1]
            label = Chars2Index(label_literal)
            labels.append(label)
    return filenames, labels
