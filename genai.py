from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
import tensorflow as tf


ds =load_dataset("Abirate/english_quotes")
data = " ".join(ds["train"]["quote"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequences = tokenizer.texts_to_sequences([data])

max_sequence_len = max([len(x) for x in sequences])
paddedd_sequences = pad_sequences(sequences, maxlen = max_sequence_len, padding = 'pre')

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length = max_sequence_len-1),
    LSTM(150),
    Dense(len(tokenizer.word_index)+1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X, y = paddedd_sequences[:, :-1], paddedd_sequences[: , -1]
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index)+1)
model.fit(X, y , epochs=1)

def generate_text(send_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([send_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0).argmax(axis=-1)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                send_text += " " + word
                break
    return send_text

print(generate_text("Once upon a time ", 50, max_sequence_len))