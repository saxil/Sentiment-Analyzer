# Sentiment Analysis Project

This project explores different approaches to sentiment analysis, progressing from a basic scikit-learn model to a more advanced BiLSTM with TensorFlow/Keras, and finally to a fine-tuned Transformer model (DistilBERT) using the Hugging Face ecosystem.

## Project Structure

```
.
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── best_model.h5         # Saved weights for the BiLSTM model (v1.py)
├── best_transformer/     # Saved model and tokenizer for the fine-tuned DistilBERT
│   ├── config.json
│   ├── tf_model.h5
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── predict.py            # Script to load the BiLSTM model (best_model.h5) and predict sentiment for custom text
├── requirements.txt      # Python dependencies
├── train_transformer.py  # Script to fine-tune and evaluate the DistilBERT model
├── v0.py                 # Basic sentiment analyzer using scikit-learn (TF-IDF + Logistic Regression)
└── v1.py                 # Advanced sentiment analyzer using TensorFlow/Keras (BiLSTM on IMDB dataset)
```

## Models

### 1. Basic Model (`v0.py`)

- **Description**: A simple sentiment analyzer using TF-IDF for feature extraction and a Logistic Regression classifier.
- **Dataset**: Small, custom dataset defined within the script.
- **Libraries**: scikit-learn, NLTK, pandas.
- **Functionality**:
    - Text preprocessing (lowercase, stopword removal, punctuation removal).
    - Trains on the custom dataset.
    - Predicts sentiment for new text.

### 2. BiLSTM Model (`v1.py`)

- **Description**: An advanced sentiment analyzer using a Bidirectional LSTM network built with TensorFlow and Keras.
- **Dataset**: IMDB movie review dataset (loaded via `tensorflow.keras.datasets.imdb`).
- **Libraries**: TensorFlow, scikit-learn, numpy.
- **Functionality**:
    - Loads and preprocesses the IMDB dataset (padding sequences).
    - Builds a BiLSTM model with Embedding, Dropout, and Dense layers.
    - Trains the model with early stopping and saves the best weights to `best_model.h5`.
    - Evaluates the model and prints a classification report.
    - Provides a function to predict sentiment for custom text.
    - Includes an interactive prompt to get user input for prediction.

### 3. Fine-tuned Transformer Model (`train_transformer.py`)

- **Description**: A state-of-the-art sentiment analyzer based on fine-tuning a pre-trained DistilBERT model from Hugging Face.
- **Dataset**: IMDB movie review dataset (loaded via `datasets` library).
- **Libraries**: TensorFlow, Hugging Face `transformers` and `datasets`, scikit-learn.
- **Functionality**:
    - Loads the IMDB dataset.
    - Tokenizes text using `AutoTokenizer` (DistilBERT).
    - Prepares `tf.data.Dataset` for training and testing.
    - Fine-tunes `TFAutoModelForSequenceClassification` (DistilBERT) on the sentiment task.
    - Evaluates the model and prints a classification report (including F1-score).
    - Saves the fine-tuned model and tokenizer to the `best_transformer/` directory.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    -   Ensure you have a compatible Python version installed (e.g., Python 3.9, 3.10, or 3.11, as TensorFlow 2.12 might have issues with newer Python versions on some platforms).
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    -   Windows (PowerShell):
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    -   Windows (Command Prompt):
        ```cmd
        .\.venv\Scripts\activate.bat
        ```
    -   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues installing TensorFlow, ensure your Python version and system architecture are supported. You might need to install a specific TensorFlow version (e.g., `tensorflow==2.11.0` if `2.12.0` causes issues).*

4.  **Download NLTK data (for `v0.py`):**
    Run Python and execute:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Running the Models

### `v0.py` (Basic Model)

```bash
python v0.py
```
Output will show accuracy, classification report, and a sample prediction.

### `v1.py` (BiLSTM Model)

1.  **Train the model (if `best_model.h5` doesn't exist or you want to retrain):**
    ```bash
    python v1.py
    ```
    This will train the model, save `best_model.h5`, print evaluation metrics, and then prompt you to enter text for sentiment prediction.

2.  **Run predictions using the saved BiLSTM model:**
    ```bash
    python predict.py
    ```
    This script loads `best_model.h5` and provides an interactive prompt for sentiment prediction.

### `train_transformer.py` (Fine-tuned DistilBERT)

1.  **Train and evaluate the Transformer model:**
    ```bash
    python train_transformer.py
    ```
    This will:
    - Download the IMDB dataset and DistilBERT model/tokenizer.
    - Fine-tune the model.
    - Print a classification report (this is where you'll see the improved F1-score).
    - Save the fine-tuned model and tokenizer to the `best_transformer/` directory.

    *Note: Training this model can take some time and may require a GPU for reasonable performance, though it will run on a CPU.*

2.  **To use the saved Transformer model for predictions:**
    You would typically write a new script similar to `predict.py` but loading the model and tokenizer from the `best_transformer/` directory using `TFAutoModelForSequenceClassification.from_pretrained('best_transformer')` and `AutoTokenizer.from_pretrained('best_transformer')`.

## Key Improvements and F1-Score

-   The `v0.py` model provides a baseline.
-   The `v1.py` (BiLSTM) model offers significantly better performance by leveraging neural networks and word embeddings.
-   The `train_transformer.py` (DistilBERT) model aims for the highest F1-score by fine-tuning a large pre-trained language model, which captures much richer contextual understanding of text. The F1-score from this model should be substantially higher than the previous two.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.
