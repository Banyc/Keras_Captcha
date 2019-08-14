import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

import glo_var
import model
import img_gen


model_train = model.Get_model()

if os.path.exists(os.path.join(glo_var.checkpoint_dir, "checkpoint")):
    model_train.load_weights(glo_var.checkpoint_path)

cp_callbacks = []

# Create checkpoint callback
cp_callbacks.append(tf.keras.callbacks.ModelCheckpoint(glo_var.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1))

# get data
x_train, y_train, x_val, y_val = img_gen.get_data()

# assemble image generator
train_data_size = len(x_train)
val_data_size = len(x_val)

train_imgGen = img_gen.ImageGen(x_train, y_train, glo_var.batch_size)
val_imgGen = img_gen.ImageGen(x_val, y_val, glo_var.batch_size)

# set off
history = model_train.fit_generator(generator=train_imgGen,
                            steps_per_epoch=int(train_data_size) // glo_var.batch_size,
                            epochs=glo_var.epoch,
                            callbacks=cp_callbacks,
                            validation_data=val_imgGen,
                            validation_steps=int(val_data_size) // glo_var.batch_size,
                            )
