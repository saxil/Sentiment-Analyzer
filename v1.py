"""Advanced Sentiment Analyzer using TensorFlow and Keras on the IMDB dataset."""
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 500
EMBEDDING_DIM = 128

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Pad sequences to uniform length
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')

# Build a Bidirectional LSTM model
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
)

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Detailed classification report
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=4))

# Prediction helper for custom text
word_index = imdb.get_word_index()
def predict_sentiment(text: str) -> str:
    '''Return sentiment for a custom text. Positive or Negative.'''
    # Tokenize and pad
    tokens = [word_index.get(word, 2) for word in text.lower().split()]
    seq = pad_sequences([tokens], maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(seq)[0][0]
    return 'Positive' if prob > 0.5 else 'Negative'

