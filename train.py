import pandas as pd
import os

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification

import numpy as np
import evaluate

import torch.nn as nn



class TextClassification(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

def main():
    
    # define lambda function for metric callback
    # eval_pred[0]: logits, eval_pred[1]: labels
    compute_metrics = lambda eval_pred: metric.compute(predictions=np.argmax(eval_pred[0], axis=-1), references=eval_pred[1])
    
    # define lambda function for input tokenization
    tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    data_dir = "datasets/reviews"

    train = os.path.join(data_dir, "b2w.csv")
    # valid = os.path.join(data_dir, "olist.csv")
    # test = os.path.join(data_dir, "utlc_movies.csv")

    train_dataset = pd.read_csv(train).dropna()

    dataset = Dataset.from_dict({"text": train_dataset['review_text_processed'].values, "label":train_dataset['polarity'].values[:, np.newaxis]})

    tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')


    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10000))


    model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", problem_type="multi_label_classification", num_labels=1)
    model = model.train()

    metric = evaluate.load("accuracy")


    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch", 
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=3e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()