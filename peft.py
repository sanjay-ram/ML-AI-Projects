import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Split dataset into training (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

# Fine-tune the model using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Start fine-tuning the model
trainer.train()

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Evaluate the model on the test set
predictions_output = trainer.predict(test_data)
predictions = predictions_output.predictions.argmax(axis=-1) # Assuming a classification task

# Compute evaluation metrics
accuracy = accuracy_score(test_data['label'], predictions)
f1 = f1_score(test_data['label'], predictions, average='weighted')
precision = precision_score(test_data['label'], predictions, average='weighted')
recall = recall_score(test_data['label'], predictions, average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test F1 Score: {f1}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")

# Use hyperparameter search to optimize fine-tuning
best_model = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10
)