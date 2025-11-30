import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

t = np.linspace(0, 100, 10000)
X= np.sin(t).reshape(-1, 1)

def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 100
X_seq, y_seq = create_sequences(X, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the RNN model
model_rnn = models.Sequential([
    layers.SimpleRNN(128, input_shape=(seq_length, 1)),
    layers.Dense(1)  # Single output for next value prediction
])

# Compile and train the model
model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

mse = model_rnn.evaluate(X_test, y_test)
print(f'Test MSE: {mse}')