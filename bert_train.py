from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# LOAD DATA

df = pd.read_csv(r"C:\Users\BHOOMI\OneDrive\Desktop\fake-job-detector\data\main_dataset.csv")
df = df.fillna("")

# 🔥 SPEED FIX (VERY IMPORTANT)
df = df.sample(3000, random_state=42)

df["text"] = df["title"] + " " + df["description"]
df["label"] = df["fraudulent"]

# TRAIN-TEST SPLIT

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# TOKENIZER

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128   # 🔥 important for speed
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# DATASET CLASS

class JobDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = JobDataset(train_encodings, train_labels)
val_dataset = JobDataset(val_encodings, val_labels)

# MODEL

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# TRAINING ARGUMENTS (FIXED)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,                 # 🔥 keep 1 for now
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# TRAIN

trainer.train()

# SAVE MODEL

model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("✅ BERT model saved!")