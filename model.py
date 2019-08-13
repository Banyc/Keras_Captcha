import tensorflow as tf
import glo_var


def Get_model():
    input_tensor = tf.keras.Input(glo_var.img_shape)
    tensor = input_tensor
    for i in range(2):
        tensor = tf.keras.layers.Conv2D(32 * 2 ** i, (3, 3), activation="relu", padding="same", data_format="channels_last")(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Conv2D(32 * 2 ** i, (3, 3), activation="relu", padding="same")(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same")(tensor)
    tensor = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(tensor)
    tensor = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(tensor)
    tensor = tf.keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(tensor)
    tensor = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same")(tensor)

    tensor = tf.keras.layers.Flatten()(tensor)

    branches = []

    for i in range(glo_var.num_classes):
        branch = tf.keras.layers.Dense(64, activation="relu")(tensor)
        branch = tf.keras.layers.Dropout(0.25)(branch)
        branch = tf.keras.layers.Dense(glo_var.num_chars, activation="softmax", name=str(i))(branch)
        branches.append(branch)

    model = tf.keras.Model(inputs=input_tensor, outputs=branches)

    # adam = tf.keras.optimizers.Adam(lr=glo_var.learning_rate, decay=glo_var.decay_rate)
    # adam = tf.keras.optimizers.Adam(lr=glo_var.learning_rate)
    adam = tf.keras.optimizers.Adam()

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'],
                )

    model.summary()
    
    return model
    
if  __name__ == "__main__": 
    model = Get_model()
