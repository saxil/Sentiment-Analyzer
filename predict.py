"""Load a trained sentiment analysis model and predict custom inputs."""
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Parameters (must match those used in training)
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Load the IMDB word index
data_word_index = imdb.get_word_index()


def predict_sentiment(text: str) -> str:
    """Return 'Positive' or 'Negative' for a given text input."""
    # Tokenize text (unknown words mapped to index 2)
    tokens = [data_word_index.get(word, 2) for word in text.lower().split()]
    # Pad/truncate sequence
    seq = pad_sequences([tokens], maxlen=MAX_LEN, padding='post', truncating='post')
    # Predict probability
    prob = model.predict(seq, verbose=0)[0][0]
    return 'Positive' if prob > 0.5 else 'Negative'


def main():
    print("Loaded model: best_model.h5")
    print("Type your text (or 'exit' to quit):")
    while True:
        user_input = input('> ').strip()
        if user_input.lower() in ('exit', 'quit'):
            print("Exiting.")
            break
        sentiment = predict_sentiment(user_input)
        print(f"Sentiment: {sentiment}\n")


if __name__ == '__main__':
    main()
