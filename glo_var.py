import os

import git_secret

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_chars = len(chars)
num_classes = 4

num_classes_total = len(chars) ** num_classes

# (height, width, num_channels)
img_shape = (64, 128, 3)

image_dir = git_secret.image_dir
TEST_DATA_DIR = git_secret.TEST_DATA_DIR

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

data_category_info_dir = checkpoint_dir
x_train_npy_filename = "x_train.npy"
y_train_npy_filename = "y_train.npy"
x_val_npy_filename = "x_val.npy"
y_val_npy_filename = "y_val.npy"

epoch = 6
# learning_rate = 0.00001
# learning_rate = 0.0001
# learning_rate = 0.001
# decay_rate = 0.90

batch_size = 64
