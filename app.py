import re
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class TweetDataset(Dataset):
    def __init__(self, texts, labels, input_ids, attention_mask):
        self.texts = texts
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'label': torch.tensor(self.labels[idx])
        }

def custom_tokenizer(texts, vocab_size=30522):
    input_ids = []
    attention_mask = []
    vocab = {}

    for text in texts:
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        sample_input_ids = [vocab.get(token, vocab_size - 1) for token in tokens]
        sample_attention_mask = [1] * len(sample_input_ids)

        input_ids.append(sample_input_ids)
        attention_mask.append(sample_attention_mask)

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    vocab_size = len(vocab) + 2  # Add 2 for special tokens (PAD and UNK)
    return input_ids, attention_mask, vocab, {idx: word for word, idx in vocab.items()}

# Load the dataset
dataset = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
val_texts = dataset['validation']['text']
val_labels = dataset['validation']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

train_input_ids, train_attention_mask, vocab, inv_vocab = custom_tokenizer(train_texts)
val_input_ids, val_attention_mask, _, _ = custom_tokenizer(val_texts)
test_input_ids, test_attention_mask, _, _ = custom_tokenizer(test_texts)

# Save the tokenizer
torch.save({'input_ids': train_input_ids, 'attention_mask': train_attention_mask, 'vocab': vocab, 'inv_vocab': inv_vocab}, './custom_tokenizer.pt')

# Prepare the data loaders
train_dataset = TweetDataset(train_texts, train_labels, train_input_ids, train_attention_mask)
val_dataset = TweetDataset(val_texts, val_labels, val_input_ids, val_attention_mask)
test_dataset = TweetDataset(test_texts, test_labels, test_input_ids, test_attention_mask)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Initialize the model
config = RobertaConfig(vocab_size=len(vocab) + 2, num_labels=3)
model = RobertaForSequenceClassification(config)

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
