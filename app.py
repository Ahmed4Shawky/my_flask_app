import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

# Load the dataset from the local directory
dataset = load_dataset('/amazon_fine_food_reviews')

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
config = RobertaConfig(num_labels=5)  # Assuming 5 classes for star ratings
model = RobertaForSequenceClassification(config)

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['Text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./custom_roberta_model')
