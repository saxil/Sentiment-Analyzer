"""Fine-tune a DistilBERT model on the IMDB dataset to improve F1 score."""
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report

# Configuration
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 1. Load IMDB dataset
raw_datasets = load_dataset('imdb')

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# 3. Preprocess
encoded = raw_datasets.map(tokenize_batch, batched=True)
encoded = encoded.remove_columns(['text'])
encoded.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label'])

# 4. Create tf.data.Dataset
train_ds = encoded['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='label',  # Changed from ['label']
    shuffle=True,
    batch_size=BATCH_SIZE
)
test_ds = encoded['test'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='label',  # Changed from ['label']
    shuffle=False,
    batch_size=BATCH_SIZE
)

# 5. Model
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

# 6. Train
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# 7. Evaluate
preds = model.predict(test_ds).logits
y_pred = tf.argmax(preds, axis=1).numpy()
y_true = raw_datasets['test']['label']
print(classification_report(y_true, y_pred, digits=4))

# 8. Save
model.save_pretrained('best_transformer')
tokenizer.save_pretrained('best_transformer')
print("Model and tokenizer saved to ./best_transformer/")
