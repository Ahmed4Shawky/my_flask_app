import re
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

# Inspect the dataset structure
for batch in dataset['train']:
    print(batch.keys())
    break

# Save the tokenizer
torch.save({'vocab': vocab, 'inv_vocab': inv_vocab}, './custom_tokenizer.pt')

# Initialize the model
config = RobertaConfig(vocab_size=len(vocab) + 2, num_labels=3)
model = RobertaForSequenceClassification(config)

# Prepare the data loaders
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Debug the DataLoader
for batch in train_dataloader:
    print(batch)
    break

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train loss: {total_train_loss / len(train_dataloader)}, Val loss: {total_val_loss / len(val_dataloader)}")

# Save the model
model.save_pretrained('./custom_roberta_model')
