import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
import tensorflow_probability as tfp

from utils import display

IMAGE_SIZE = 32
N_COMPONENTS = 5
EPOCHS = 10
BATCH_SIZE = 128

(x_train, _), (_, _) = datasets.fashion_mnist.load_data()


def preprocess(imgs):
    imgs = np.expand_dims(imgs, -1)
    imgs = tf.image.resize(imgs, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return imgs

input_data = preprocess(x_train)

dist = tfp.distributions.PixelCNN(
    image_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=N_COMPONENTS,
    dropout_p=0.3
)

image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
log_prob = dist.log_prob(image_input)
pixelcnn = models.Model(inputs=image_input, outputs=log_prob)
pixelcnn.add_loss(-tf.reduce_mean(log_prob))

pixelcnn.compile(optimizer=optimizers.Adam(0.001))

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img
    
    def generate(self):
        return dist.sample(self.num_img).numpy()
    
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate()
        display(
            generated_images,
            n=self.num_img,
            save_to="./output_3/generated_img_%03d.png" % (epoch),
        )

img_generator_callback = ImageGenerator(num_img=2)
pixelcnn.fit(
    input_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=True,
    callbacks=[img_generator_callback]
)

generated_images = img_generator_callback.generate()
display(generated_images, n=img_generator_callback.num_img, save_to='./output_3/gen_img.png')