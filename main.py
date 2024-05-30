import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices())

dataset_name = 'data.csv'
# Load the CSV data
df = pd.read_csv(dataset_name, sep=';', encoding='latin1')

# Tokenization and padding
prompt_tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
prompt_tokenizer.fit_on_texts(df['prompt'].tolist())

input_sequences = prompt_tokenizer.texts_to_sequences(df['prompt'].tolist())


script_tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
script_tokenizer.fit_on_texts(df['gdscript_code'].tolist())

target_sequences = script_tokenizer.texts_to_sequences(df['gdscript_code'].tolist())

max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)
max_len = max(max_input_len, max_target_len)

input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

input_vocab_size = len(prompt_tokenizer.word_index) + 1
target_vocab_size = len(script_tokenizer.word_index) + 1


# Define the split ratio
split_ratio = 0.2
indices = np.arange(input_sequences.shape[0])
np.random.shuffle(indices)

split_index = int((1 - split_ratio) * input_sequences.shape[0])

train_indices = indices[:split_index]
val_indices = indices[split_index:]

X_train = input_sequences[train_indices]
y_train = target_sequences[train_indices]
X_val = input_sequences[val_indices]
y_val = target_sequences[val_indices]

def create_dataset(inputs, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(buffer_size=len(inputs))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 1

# Create dataset with inputs and targets correctly shaped
train_dataset = create_dataset((X_train, y_train[:, :-1]), y_train[:, 1:], batch_size)
val_dataset = create_dataset((X_val, y_val[:, :-1]), y_val[:, 1:], batch_size)



def get_positional_encoding(max_seq_len, dm):
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def transformer_decoder(inputs, enc_outputs, head_size, num_heads, ff_dim, dropout=0):
    causal_mask = tf.linalg.band_part(tf.ones((inputs.shape[1], inputs.shape[1])), -1, 0)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, attention_mask=causal_mask)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, enc_outputs)
    x = layers.Dropout(dropout)(x)
    res = x + res

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer(input_vocab_size, target_vocab_size, input_length, target_length, head_size, num_heads, ff_dim, num_transformer_blocks, dropout=0):
    inputs = layers.Input(shape=(input_length,))
    enc_embedding = layers.Embedding(input_vocab_size, head_size)(inputs)
    enc_positional_encoding = get_positional_encoding(input_length, head_size)
    enc_outputs = enc_embedding + enc_positional_encoding

    for _ in range(num_transformer_blocks):
        enc_outputs = transformer_encoder(enc_outputs, head_size, num_heads, ff_dim, dropout)

    dec_inputs = layers.Input(shape=(target_length,))
    dec_embedding = layers.Embedding(target_vocab_size, head_size)(dec_inputs)
    dec_positional_encoding = get_positional_encoding(target_length, head_size)
    dec_outputs = dec_embedding + dec_positional_encoding

    for _ in range(num_transformer_blocks):
        dec_outputs = transformer_decoder(dec_outputs, enc_outputs, head_size, num_heads, ff_dim, dropout)

    outputs = layers.Dense(target_vocab_size, activation="softmax")(dec_outputs)
    return Model([inputs, dec_inputs], outputs)

input_vocab_size = input_vocab_size
target_vocab_size = target_vocab_size
input_length = max_input_len
target_length = max_target_len - 1  # Shifted target length
head_size = 1
num_heads = 2
ff_dim = 4
num_transformer_blocks = 10
dropout = 0.2

transformer = build_transformer(input_vocab_size, target_vocab_size, input_length, target_length, head_size, num_heads, ff_dim, num_transformer_blocks, dropout)
transformer.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
transformer.summary()



transformer.fit(train_dataset, epochs=1, validation_data=val_dataset)

transformer.save('model.keras')

# Define a function to generate code for a single input prompt
def generate_code(prompt):
    # Tokenize the prompt
    input_seq = prompt_tokenizer.texts_to_sequences([prompt])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # Generate code using the model
    output_seq = np.zeros((1, max_target_len - 1))  # Placeholder for output sequence
    for i in range(max_target_len - 1):
        # Predict the next token
        predictions = transformer.predict([input_seq, output_seq])
        predicted_token_index = np.argmax(predictions[:, i, :], axis=-1)
        output_seq[0, i] = predicted_token_index

    # Decode the output sequence
    generated_code = script_tokenizer.sequences_to_texts(output_seq)[0]
    return generated_code

# Test the model with a sample prompt
prompt = "generate a function, that adds two numbers"
generated_code = generate_code(prompt)
print("Generated Code:")
print(generated_code)

print("Alles okay :)")
