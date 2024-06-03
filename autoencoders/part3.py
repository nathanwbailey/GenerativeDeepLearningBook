import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    utils,
    metrics,
    losses,
    optimizers,
)

from scipy.stats import norm
import pandas as pd

from notebooks.utils import sample_batch, display

from vae_utils import get_vector_from_label, add_vector_to_images, morph_faces

IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 128
NUM_FEATURES = 128
Z_DIM = 200
LEARNING_RATE = 0.0005
EPOCHS = 10
BETA = 2000
LOAD_MODEL = False

train_data = utils.image_dataset_from_directory(
    "",
    labels=None,
    color_mode="rgb",
    image_size=(64,64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear"
)

def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train_data = train_data.map(lambda x: preprocess(x))

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)