from flask import Flask, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Train a quick LSTM Music Generation Model upon starting
vocab_size = 10
seq_length = 20

np.random.seed(42)
raw_data = np.random.randint(0, vocab_size, size=500)
X, y = [], []
for i in range(len(raw_data) - seq_length):
    X.append(raw_data[i:i + seq_length])
    y.append(raw_data[i + seq_length])

X = np.array(X)
y = to_categorical(y, num_classes=vocab_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) / float(vocab_size)

model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate')
def generate():
    start_seed = np.random.randint(0, len(X))
    pattern = X[start_seed].flatten()
    generated_notes = []
    
    for _ in range(12):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = int(np.argmax(prediction))
        generated_notes.append(index)
        pattern = np.append(pattern[1:], index / float(vocab_size))
        
    return jsonify({"notes": generated_notes})

if __name__ == '__main__':
    app.run(debug=True)
