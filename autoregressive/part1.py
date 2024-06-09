import numpy as np
import json
import re
import string

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses

VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
N_UNITS = 128
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 25

with open('../data/recipes/full_format_recipes.json') as json_data:
    recipe_data = json.load(json_data)


filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None 
]

n_recipes = len(filtered_data)

def pad_punctuation(s):
    #Find a punctuation char and pad with spaces
    s = re.sub(f"([{string.punctuation}])",  r" \1 ", s)
    #Replace any sequence of one or more spaces with a single space
    s = re.sub(" +", " ", s)
    return s

text_data = [pad_punctuation(x) for x in filtered_data]
print(len(text_data))
print(text_data[3])

text_ds = (
    tf.data.Dataset.from_tensor_slices(text_data).batch(BATCH_SIZE).shuffle(1000)
)

vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN+1
)

vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

for i, word in enumerate(vocab[:10]):
    print(f"{i}: {word}")

print(text_data[3])
print(vectorize_layer(text_data[3]).numpy())

# Our LSTM based model takes in one token at a time and outputs the probaility of the next token
# So we shift the data, at time step 0 we feed in token 0, and LSTm model preidtcs what token 1 will be
# Etc Etc
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

train_ds = text_ds.map(prepare_inputs)

for data in text_ds.unbatch().take(5):
    print(data)
    x, y = prepare_inputs(data)
    print(x)
    print(y)


inputs = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x = layers.LSTM(128, return_sequences=True)(x)
outputs = layers.Dense(VOCAB_SIZE, activation='softmax')(x)

lstm = models.Model(inputs, outputs)
lstm.summary()

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile("adam", loss_fn)

class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word):
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}
    
    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        # Sample an index using the probability distribution
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({'prompt': start_prompt, 'word_probs': probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + ' ' + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)


text_generator = TextGenerator(vocab)
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)
lstm.fit(train_ds, epochs=EPOCHS, callbacks=[model_checkpoint_callback, text_generator])
lstm.save("./models/lstm")

def print_probs(info, vocab, top_k=5):
    for i in info:
        print(f"\nPROMPT: {i['prompt']}")
        word_probs = i["word_probs"]
        #Reverse the sequence
        # So get the top_k word_probs
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")

info = text_generator.generate("recipe for roasted vegetables | chop 1 /", max_tokens=10, temperature=1.0)
print_probs(info, vocab)