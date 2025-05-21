import tensorflow as tf
from tensorflow.keras import layers

def unet_colorizer(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D(2)(c2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    u1 = layers.UpSampling2D(2)(b)
    concat1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    u2 = layers.UpSampling2D(2)(c3)
    concat2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
