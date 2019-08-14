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


#region "image reform"
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
    image = Image.open(imagePath)
    # size := (width, height)
    image = image.resize((glo_var.img_shape[1], glo_var.img_shape[0]))
    image_arr = np.array(image)
    image_arr = image_arr.astype(np.float32)
    image_arr = image_arr - 147.0
    # alternative tunning
    # image_arr = image_arr / 255.0
    # image_arr = image_arr - 0.5

    return image_arr    
#endregion


#region "Scan dir and get all images and labels for train"
def __get_filenames_labels():
    labels = []
    for _, _, filenames in os.walk(glo_var.image_dir):
        for filename in filenames:
            label_literal = filename.split('.')[0].split('_')[-1]
            label = Chars2Index(label_literal)
            labels.append(label)
    return filenames, labels


def __split_data(filenames, labels, ratio=0.3):
    import sklearn
    # shuffle
    filenames, labels = sklearn.utils.shuffle(filenames, labels)
    # split dataset
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        filenames,
        labels,
        test_size=ratio,
    )
    return (x_train, y_train, x_val, y_val)


def get_data(ratio=0.3):
    if not os.path.exists(os.path.join(glo_var.data_category_info_dir)):
        os.makedirs(glo_var.data_category_info_dir)
    if not os.path.exists(os.path.join(glo_var.data_category_info_dir, glo_var.x_train_npy_filename)):
        filenames, labels = __get_filenames_labels()
        x_train, y_train, x_val, y_val = __split_data(filenames, labels, ratio)
        np.save(os.path.join(glo_var.data_category_info_dir, glo_var.x_train_npy_filename), x_train)
        np.save(os.path.join(glo_var.data_category_info_dir, glo_var.y_train_npy_filename), y_train)
        np.save(os.path.join(glo_var.data_category_info_dir, glo_var.x_val_npy_filename), x_val)
        np.save(os.path.join(glo_var.data_category_info_dir, glo_var.y_val_npy_filename), y_val)
    else:
        x_train = np.load(os.path.join(glo_var.data_category_info_dir, glo_var.x_train_npy_filename))
        y_train = np.load(os.path.join(glo_var.data_category_info_dir, glo_var.y_train_npy_filename))
        x_val = np.load(os.path.join(glo_var.data_category_info_dir, glo_var.x_val_npy_filename))
        y_val = np.load(os.path.join(glo_var.data_category_info_dir, glo_var.y_val_npy_filename))
    return (list(x_train), list(y_train), list(x_val), list(y_val))
    # return (x_train, y_train, x_val, y_val)
#endregion


if __name__ == "__main__":
    get_data()
