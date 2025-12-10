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

from lora import LoRALayer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load a pre-trained BERT model for classification tasks
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Print model layers to identify attention layers where LoRA can be applied
for name, module in model.named_modules():
    print(name)  # This output helps locate attention layers

# Apply LoRA to attention layers
for name, module in model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)

# Freeze other layers to update only LoRA-modified parameters
for param in model.base_model.parameters():
    param.requires_grad = False



# Configure training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
)

# Set up the Trainer to handle fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Begin training
trainer.train()

# Evaluate the LoRA fine-tuned model on the test set
results = trainer.evaluate(eval_dataset=test_data)
print(f"Test Accuracy: {results['eval_accuracy']}")

# Example of adjusting the rank in LoRA
from lora import adjust_lora_rank

# Set a lower rank for fine-tuning, experiment with values for optimal performance
adjust_lora_rank(model, rank=2)