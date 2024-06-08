import re
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Custom Tokenizer
def custom_tokenizer(text, vocab_size=30522):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, {idx: word for word, idx in vocab.items()}

# Load the dataset
dataset = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
texts = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']
vocab, inv_vocab = custom_tokenizer(" ".join(texts))

# Save the tokenizer
torch.save({'vocab': vocab, 'inv_vocab': inv_vocab}, './custom_tokenizer.pt')

# Initialize the model
config = RobertaConfig(vocab_size=len(vocab) + 2, num_labels=3)
model = RobertaForSequenceClassification(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    run_name=None  # Disable wandb integration
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./custom_roberta_model')
