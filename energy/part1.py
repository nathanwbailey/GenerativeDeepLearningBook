import numpy as np

import tensorflow as tf
from tensorflow.keras import (
    datasets,
    layers,
    models,
    optimizers,
    activations,
    metrics,
    callbacks,
)

from utils import display
import random

IMAGE_SIZE = 32
CHANNELS = 1
STEP_SIZE = 10
STEPS = 60
NOISE = 0.005
ALPHA = 0.1
GRADIENT_CLIP = 0.03
BATCH_SIZE = 128
BUFFER_SIZE = 8192
LEARNING_RATE = 0.0001
EPOCHS = 60
LOAD_MODEL = False

(x_train, _), (x_test, _) = datasets.mnist.load_data()

def preprocess(imgs):
    imgs = (imgs.astype("float32") - 127.5) / 127.5
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=1.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

x_train = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE)
x_test = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)


ebm_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
x = layers.Conv2D(16, kernel_size=5, strides=2, padding="same", activation=activations.swish)(ebm_input)
x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation=activations.swish)(x)
ebm_output = layers.Dense(1)(x)
model = models.Model(ebm_input, ebm_output)
model.summary()

if LOAD_MODEL:
    model.load_weights("./models/model.h5")

def generate_samples(model, input_imgs, steps, step_size, noise, return_img_per_step=False):
    imgs_per_step = []
    for _ in range(steps):
        #Add some noise to the images
        input_imgs += tf.random.normal(input_imgs.shape, mean = 0, stddev=noise)
        input_imgs = tf.clip_by_value(input_imgs, -1.0, 1.0)
        with tf.GradientTape() as tape:
            #input_imgs are not parameters of the model, so we must watch them
            tape.watch(input_imgs)
            out_score = model(input_imgs)
        grads = tape.gradient(out_score, input_imgs)
        grads = tf.clip_by_value(grads, -GRADIENT_CLIP, GRADIENT_CLIP)
        input_imgs += step_size * grads
        input_imgs = tf.clip_by_value(input_imgs, -1.0, 1.0)
        if return_img_per_step:
            imgs_per_step.append(input_imgs)
    if return_img_per_step:
        return tf.stack(imgs_per_step, axis=0)
    return input_imgs

class Buffer:
    def __init__(self, model):
        self.model = model
        #Buffer init with random noise [-1, 1] range
        self.examples = [
            tf.random.uniform(shape=(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) * 2 - 1 for _ in range(BATCH_SIZE)
        ]
    
    def sample_new_exmps(self, steps, step_size, noise):
        # 5% of the observations are random on average
        # For each in BATCH SIZE, you either sample with prob 0.05 or you don't
        # BATCH SIZE number of bernoulli variables with prob 0.05
        # Get a number between 0 and 32, mean is 32*0.05 = 1.6
        # So on average 5% of samples are random
        n_new = np.random.binomial(BATCH_SIZE, 0.05)
        rand_imgs = (tf.random.uniform(shape=(n_new, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) * 2 - 1)
        old_imgs = tf.concat(random.choices(self.examples, k=BATCH_SIZE - n_new), axis=0)
        input_imgs = tf.concat([rand_imgs, old_imgs], axis=0)
        input_imgs = generate_samples(self.model, input_imgs, steps=steps, step_size=step_size, noise=noise)
        #Add generated examples to the buffer
        self.examples = tf.split(input_imgs, BATCH_SIZE, axis=0) + self.examples
        #Limit the size of the buffer
        self.examples = self.examples[:BUFFER_SIZE]
        return input_imgs

class EBM(models.Model):
    def __init__(self):
        super().__init__()
        self.model = model
        self.buffer = Buffer(self.model)
        self.alpha = ALPHA
        self.loss_metric = metrics.Mean(name="loss")
        self.reg_loss_metric = metrics.Mean(name="reg")
        self.cdiv_loss_metric = metrics.Mean(name="cdiv")
        self.real_out_metric = metrics.Mean(name="real")
        self.fake_out_metric = metrics.Mean(name="fake")
    
    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.reg_loss_metric,
            self.cdiv_loss_metric,
            self.real_out_metric,
            self.fake_out_metric,
        ]

    def train_step(self, real_imgs):
        real_imgs += tf.random.normal(
            shape=tf.shape(real_imgs), mean=0, stddev=NOISE
        )
        real_imgs = tf.clip_by_value(real_imgs, -1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(steps=STEPS, step_size=STEP_SIZE, noise=NOISE)
        inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
        with tf.GradientTape() as training_tape:
            #Split the output, as we have BATCH_SIZE real and BATCH_SIZE fake, this will give us the real and fake outputs
            real_out, fake_out = tf.split(self.model(inp_imgs), 2, axis=0)
            cdiv_loss = tf.reduce_mean(fake_out, axis=0) - tf.reduce_mean(real_out, axis=0)
            reg_loss = self.alpha * tf.reduce_mean(real_out**2 + fake_out**2, axis=0)
            loss = cdiv_loss + reg_loss
        grads = training_tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        self.reg_loss_metric.update_state(reg_loss)
        self.cdiv_loss_metric.update_state(cdiv_loss)
        self.real_out_metric.update_state(tf.reduce_mean(real_out, axis=0))
        self.fake_out_metric.update_state(tf.reduce_mean(fake_out, axis=0))
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = (
            tf.random.uniform((batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) * 2 -1
        )
        inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
        real_out, fake_out = tf.split(self.model(inp_imgs), 2)
        cdiv_loss = tf.reduce_mean(fake_out, axis=0) - tf.reduce_mean(real_out, axis=0)
        self.cdiv_loss_metric.update_state(cdiv_loss)
        self.real_out_metric.update_state(tf.reduce_mean(real_out, axis=0))
        self.fake_out_metric.update_state(tf.reduce_mean(fake_out, axis=0))
        return {m.name: m.result() for m in self.metrics[2:]}
    

ebm_model = EBM()
ebm_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), run_eagerly=True)


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img
    
    def on_epoch_end(self, epoch, logs=None):
        start_imgs = (
            np.random.uniform(size=(self.num_ims, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) * 2 -1
        )
        generated_images = generate_samples(ebm_model.model, start_imgs, steps=1000, step_size=STEP_SIZE, noise=NOISE, return_img_per_step=False)
        generated_images = generated_images.numpy()
        display(
            generated_images,
            save_to="./output/generated_img_%03d.png" % (epoch),
        )
        example_images = tf.concat(
            random.choices(ebm_model.buffer.examples, k=10), axis=0
        )
        example_images = example_images.numpy()
        display(
            example_images, save_to="./output/example_img_%03d.png" % (epoch)
        )

image_generator_callback = ImageGenerator(num_img=10)
ebm_model.fit(
    x_train,
    shuffle=True,
    epochs=60,
    validation_data=x_test,
    callbacks=[
        image_generator_callback,
    ],
    verbose=2
)

start_imgs = (
    np.random.uniform(size=(10, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)) * 2 - 1
)
display(start_imgs, save_to='./output/start_imgs.png')
gen_img = generate_samples(
    ebm_model.model,
    start_imgs,
    steps=1000,
    step_size=STEP_SIZE,
    noise=NOISE,
    return_img_per_step=False
)
display(gen_img[-1].numpy(), save_to='./output/end_imgs.png')
imgs = []
for i in [0, 1, 3, 5, 10, 30, 50, 100, 300, 999]:
    imgs.append(gen_img[i].numpy()[6])

display(np.array(imgs), save_to='./output/end_imgs_progress.png')

