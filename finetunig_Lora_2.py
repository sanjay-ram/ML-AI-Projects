import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('my_dataset.csv')

train_data, temp_data = train_test_split(df, test_size=0.5, random_state=42)
val_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

from transformers import BertForNextSentencePrediction, TrainingArguments, Trainer
from lora import LoRALayer

model = BertForNextSentencePrediction('bert-base-uncased', num_labels=3)

for name, module in model.named_modules():
    print(name)

for name, module in model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

for param in model.base_model.parameters():
    param.requires_grad = False

training_args = TrainingArguments(
    output_dir= './result',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy= "epoch",
)

trainer = Trainer(
    model = model,
    args= training_args,
    train_dataset= train_data,
    eval_dataset= val_data,
)

trainer.train()

result = trainer.evaluate(eval_dataset=val_data)
print(f"Accuracy: {result['accuracy']}")

from lora import adjust_lora_rank

adjust_lora_rank(model, rank=4)

alpha = 16
dropout_rate = 0.1
use_bias = True

if hasattr(model.config, 'alpha'):
    model.config.alpha = alpha

else:
    print("Warning: model.config does not have attribute 'alpha'")


if hasattr(model.config, 'hidden_dropout_prob'):
    model.config.hidden_dropout_prob = dropout_rate

else:
    print("Warning: model.config does not have attribute 'hidden_dropout_prob'")

if hasattr(model.config, 'use_bias'):
    model.config.use_bias  = use_bias

else:
    print("Warning: model.config does not have attribute 'use_bias'")