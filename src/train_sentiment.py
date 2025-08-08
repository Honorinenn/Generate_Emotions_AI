"""Fine-tune BERT for sentiment classification using Hugging Face Trainer."""
import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
DATASET = os.getenv("DATASET", "glue")  # default: glue:sst2


def preprocess(examples, tokenizer):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    return {"accuracy": acc}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Load SST-2
    ds = load_dataset('glue', 'sst2')
    ds = ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    ds = ds.remove_columns([c for c in ds['train'].column_names if c not in ['input_ids','attention_mask','label']])
    ds.set_format(type='torch')

    args = TrainingArguments(
        output_dir='outputs/models/bert-sentiment',
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()