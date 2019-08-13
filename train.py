import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

import glo_var
import model
import img_gen


model_train = model.Get_model()

if os.path.exists(glo_var.checkpoint_dir):
    model_train.load_weights(glo_var.checkpoint_path)

cp_callbacks = []

# Create checkpoint callback
cp_callbacks.append(tf.keras.callbacks.ModelCheckpoint(glo_var.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1))

filenames, labels = img_gen.get_filenames_labels()

data_size = len(filenames)

imgGen = img_gen.ImageGen(filenames, labels, glo_var.batch_size)

history = model_train.fit_generator(generator=imgGen,
                            steps_per_epoch=int(data_size) // glo_var.batch_size,
                            epochs=glo_var.epoch,
                            callbacks=cp_callbacks,
                            )
