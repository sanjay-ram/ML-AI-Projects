import re
import pandas as pd

# Create a noisy sample dataset
data_dict = {
    "text": [
        "  The staff was very kind and attentive to my needs!!!  ",
        "The waiting time was too long, and the staff was rude. Visit us at http://hospitalreviews.com",
        "The doctor answered all my questions...but the facility was outdated.   ",
        "The nurse was compassionate & made me feel comfortable!! :) ",
        "I had to wait over an hour before being seen.  Unacceptable service! #frustrated",
        "The check-in process was smooth, but the doctor seemed rushed. Visit https://feedback.com",
        "Everyone I interacted with was professional and helpful. 😊  "
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive"]
}

# Convert to a DataFrame
data = pd.DataFrame(data_dict)

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

# Apply the cleaning function
data['cleaned_text'] = data['text'].apply(clean_text)
print(data[['cleaned_text', 'label']].head())

from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the cleaned text
def tokenize_function(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Apply tokenization
data['tokenized'] = data['cleaned_text'].apply(tokenize_function)
print(data[['tokenized', 'label']].head())

# Check for missing data
print(data.isnull().sum())

# Option 1: Drop rows with missing data
data = data.dropna()

# Option 2: Fill missing values with a placeholder
data['cleaned_text'].fillna('missing', inplace=True)

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Collect input IDs and attention masks from tokenized data
input_ids_list = [token['input_ids'].squeeze() for token in data['tokenized']]
attention_masks_list = [token['attention_mask'].squeeze() for token in data['tokenized']]

# Apply padding so all sequences have the same length
input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)

# Convert labels into numeric classes (0 = negative, 1 = neutral, 2 = positive)
labels = torch.tensor([
    0 if label == "negative" else 1 if label == "neutral" else 2
    for label in data['label']
], dtype=torch.long)

# Create dataset and dataloader for training
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("DataLoader created successfully!")

from sklearn.model_selection import train_test_split

# Split data into training, validation, and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42
)

# Create DataLoader objects
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("Data splitting successful!")