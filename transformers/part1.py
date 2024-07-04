import json
import re
import string

import numpy as np
import tensorflow as tf
from IPython.display import HTML, display
from tensorflow.keras import callbacks, layers, losses, models

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 20


with open(
    "../data/wine-reviews/winemag-data-130k-v2.json", encoding="utf-8"
) as json_data:
    wine_data = json.load(json_data)

# print(wine_data[10])

# filtered_data = [
#     f"wine_review : {x['country']} : "
#     f"{x['province']} : {x['variety']} : {x['description']}"
#     for x in wine_data
#     if all(
#         [
#             x.get("country"),
#             x.get("province"),
#             x.get("variety"),
#             x.get("description"),
#         ]
#     )
# ]

filtered_data = [
    f"wine_review : {country} : {province} : {variety} : {description}"
    for x in wine_data
    if all(
        (
            country := x.get("country"),
            province := x.get("province"),
            variety := x.get("variety"),
            description := x.get("description"),
        )
    )
]


n_wines = len(filtered_data)
print(f"{n_wines} recipes loaded")

print(filtered_data[10])


def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

text_dataset = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .batch(BATCH_SIZE)
    .shuffle(1000)
)

vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN + 1,
)

vectorize_layer.adapt(text_dataset)
vocab = vectorize_layer.get_vocabulary()

assert len(vocab) <= VOCAB_SIZE

print(vectorize_layer(filtered_data[10]).numpy())


def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_dataset = text_dataset.map(prepare_inputs)

example_input_output = train_dataset.take(1).get_single_element()
# print(example_input_output)


# Causal Attention Mask
# In the form (key_length, query_length)
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    # Create 2 tensors providing integers in the range 0-n_dest/n_src
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    # Create a mask of size (n_dest, n_src)
    # (i, j) is true if i >= j - n_src + n_dest
    mask = i >= j - n_src + n_dest
    # Cast the mask to the type
    mask = tf.cast(mask, dtype)
    # Reshape the mask to add a dimension
    mask = tf.reshape(mask, [1, n_dest, n_src])
    # Gives a tensor [batch_size, 1, 1]
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
        0,
    )
    # Tile the mask to produce the size (batch_size, n_dest, n_src)
    # Replicates mask mult times
    # Repeats the mask batch_size number of times across the first axis
    # E.g [mask, mask, mask ...]
    return tf.tile(mask, mult)


x = causal_attention_mask(2, 10, 10, dtype=tf.int32)

# Create the transformer


class TransformerBlock(layers.Layer):
    def __init__(
        self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, dtype=tf.bool
        )
        attention_output, attention_scores = self.attn(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return self.ln_2(out1 + ffn_output), attention_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_len, output_dim=embed_dim
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


# Transformer Model
inputs = layers.Input(shape=(None,), dtype=tf.int32)
x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x, attention_scores = TransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM
)(x)
outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)

gpt_model = models.Model(inputs=inputs, outputs=[outputs, attention_scores])

gpt_model.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), None])
gpt_model.summary()


class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temp):
        probs = probs ** (1 / temp)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temp):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        # 0 is stop token
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temp)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    # Get the attention scores for the prediction
                    # -1 as we get an attention score
                    # if each word was used as a query
                    # So we want the last one
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            start_prompt += f" {self.index_to_word[sample_token]}"
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("wine review", max_tokens=80, temp=1.0)


text_generator = TextGenerator(vocab)
gpt_model.fit(train_dataset, epochs=EPOCHS, callbacks=[text_generator])


def print_probs(info, vocab, top_k=5):
    for i in info:
        highlighted_text = []
        for word, att_score in zip(
            i["prompt"].split(), np.mean(i["atts"], axis=0)
        ):
            highlighted_text.append(
                '<span style="background-color:rgba(135,206,250,'
                + str(att_score / max(np.mean(i["atts"], axis=0)))
                + ');">'
                + word
                + "</span>"
            )
        highlighted_text = " ".join(highlighted_text)
        display(HTML(highlighted_text))

        word_probs = i["word_probs"]
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")


info = text_generator.generate("wine review : us", max_tokens=80, temp=1.0)

info = text_generator.generate("wine review : italy", max_tokens=80, temp=0.5)

info = text_generator.generate(
    "wine review : germany", max_tokens=80, temp=0.5
)
print_probs(info, vocab)
