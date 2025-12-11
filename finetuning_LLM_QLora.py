import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('my_data.csv')

train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

from transformers import GPT2ForSequenceClassification, TrainingArguments, Trainer
from lora import LoRALayer, QuantizeModel

model = GPT2ForSequenceClassification.from_pretrained('gpt2')

quantized_model = QuantizeModel(model, bits=8)

for name, module in quantized_model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

for param in quantized_model.base_model.parameters():
    param.requires_grad = False


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(
    model=quantized_model,
    args= training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

results = Trainer.evaluate(eval_dataset=val_data)
print(f"Test Accuracy: {results['eval_accuracy']}")

from qlora import adjust_qlora_rank

adjust_qlora_rank(quantized_model, rank=4)
