import pandas as pd
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet
import random

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

data = pd.DataFrame(data_dict)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
print(data[['cleaned_text', 'label']].head())


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)

data['tokenized'] = data['cleaned_text'].apply(tokenize_function)
print(data[['tokenized', 'label']].head())

print(data.isnull().sum())
data = data.dropna()
data['cleaned_text'].fillna('missing', inplace=True)

nltk.download('wordnet')
def synom_replacement(word):
    synonyms = wordnet.synets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return word

def augment_text(text):
    words = text.split()
    augmented_words = [synom_replacement(word) if random.random() > 0.8 else word for word in words]
    return ' '.join(augmented_words)

data['augmented_text'] = data['cleaned_text'].apply(augment_text)

input_ids_list = [token['input_ids'].squeeze() for token in data['tokenized']]
attention_masks_list = [token['attention_mask'].squeeze() for token in data['tokenized']]

input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)

labels = torch.tensor([
    0 if label == 'negative' else 1 if label == 'neutral' else 2
    for label in data['label']
], dtype=torch.long)

dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("DataLoader created successfully!")

train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Train and test DataLoaders created successfully!")