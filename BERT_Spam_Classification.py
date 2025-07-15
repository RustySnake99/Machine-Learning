import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(prediction):
    labels = prediction.label_ids
    preds = prediction.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1":f1}

data = pd.read_csv("Models and Datasets/SMS_Spam_Dataset.csv", sep="\t", header=None, names=['label','message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

tokenizer = BertTokenizer.from_pretrained("bert-case-uncased")
train_txt, val_txt, train_labels, val_labels = train_test_split(data['message'].tolist(), test_size=0.2, random_state=42)

train_encodings = tokenizer(train_txt, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_txt, truncation=True, padding=True, max_length=128)

class SMS_Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SMS_Dataset(train_encodings, train_labels)
val_dataset = SMS_Dataset(val_encodings, val_labels)
model = BertForSequenceClassification.from_pretrained("bert-case-uncased", num_labels=2)

training_args = TrainingArguments(output_dir="Models and Datasets", num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16,
                                  evaluation_strategy="epoch", save_strategy="epoch", logging_dir="Models and Datasets", logging_steps=10, load_best_model_at=True,
                                  metric_for_best_model="accuracy")
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)
model.save_pretrained("Models and Datasets/bert_spam_model")
tokenizer.save_pretrained("Models and Datasets/bert_spam_model")