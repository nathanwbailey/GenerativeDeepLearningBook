from tensorflow.keras import datasets
from tensorflow import keras
import numpy as np
from utils import display
import matplotlib.pyplot as plt

(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

encoder_input = keras.layers.Input(shape=(32,32,1), name="encoder_input")
x = keras.layers.Conv2D(32, (3,3), strides=2, activation='relu', padding='same')(encoder_input)
x = keras.layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same')(x)
x = keras.layers.Conv2D(128, (3,3), strides=2, activation='relu', padding='same')(x)
shape_before_flattening = keras.backend.int_shape(x)[1:]
x = keras.layers.Flatten()(x)
encoder_output = keras.layers.Dense(2, name="encoder_output")(x)

encoder = keras.models.Model(encoder_input, encoder_output)

decoder_input = keras.layers.Input(shape=(2,), name='decoder_input')
x = keras.layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = keras.layers.Reshape(shape_before_flattening)(x)
x = keras.layers.Conv2DTranspose(128, (3,3), strides=2, activation='relu', padding="same")(x)
x = keras.layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding="same")(x)
x = keras.layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding="same")(x)
decoder_output = keras.layers.Conv2D(1, (3,3), strides=1, activation='sigmoid', padding='same', name='decoder_output')(x)

decoder = keras.models.Model(decoder_input, decoder_output)

autoencoder = keras.models.Model(encoder_input, decoder(encoder_output))
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary(expand_nested=True)

autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]
predictions = autoencoder.predict(example_images)
print("Example real clothing items")
display(example_images, save_to="examples")
print("Reconstructions")
display(predictions, save_to="reconstructions")

embeddings = encoder.predict(example_images)
plt.figure(figsize=(8,8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c = "black", alpha=0.5, s=3)
plt.show()


plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0],embeddings[:, 1],cmap="rainbow",c=example_labels, alpha=0.8,s=3)
plt.colorbar()
plt.show()

mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18,2))
reconstructions = decoder.predict(sample)

figsize = 8
grid_width, grid_height = (6, 3)
plt.figure(figsize=(figsize, figsize))

plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)

plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

fig = plt.figure(figsize=(figsize, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        str(np.round(sample[i, :], 1)),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(reconstructions[i, :, :], cmap="Greys")


figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    cmap="rainbow",
    c=example_labels,
    alpha=0.8,
    s=300,
)
plt.colorbar()

x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

reconstructions = decoder.predict(grid)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :], cmap="Greys")