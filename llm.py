import re
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

data_dict = {
    "cleaned_text": [
        "I love this product! It's amazing.",
        "This is the worst service I've ever had.",
        "It's okay, not great but not terrible.",
        "Absolutely fantastic experience, will come back again!",
        "I hate waiting in long lines."
    ],
    "label": [ "positive", "negative", "neutral", "positive", "negative" ]
}
data = pd.DataFrame(data_dict)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['cleaned_text'] = data['cleaned_text'].apply(clean_text)
print(data[['cleaned_text', 'label']].head())

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
data['tokenized'] = data['cleaned_text'].apply(tokenize_function)
print(data[['tokenized', 'label']].head())
print(data.isnull().sum())
data = data.dropna()
data['cleaned_text'].fillna('missing', inplace=True)
input_ids_list = [token['input_ids'].squeeze() for token in data['tokenized']]
attention_masks_list = [token['attention_mask'].squeeze() for token in data['tokenized']]
input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
labels = torch.tensor([
    0 if label == "negative" else 1 if label == "neutral" else 2
    for label in data['label']
], dtype=torch.long)

dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("DataLoader created successfully!")

x_train, x_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Train and test DataLoaders created successfully!")