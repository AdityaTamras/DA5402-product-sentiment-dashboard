import pandas as pd
import mlflow
import mlflow.transformers
import os
import logging
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI='http://127.0.0.1:5000'
EXPERIMENT='sentiment-classification'
MODEL_NAME='distilbert-base-uncased'
OUTPUT_DIR='models/distilbert_finetuned'
MAX_LEN=128
BATCH_SIZE=16   
EPOCHS=2

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings=tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels=labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        item={k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[index])
        return item
    
def compute_metrics(eval_pred):
    preds, y_true = eval_pred
    y_pred=np.argmax(preds, axis=1)
    macro_f1=f1_score(y_true, y_pred, average='macro')
    return {'macro_f1': macro_f1}

def train():
    train_df=pd.read_csv('data/processed/train.csv')
    test_df=pd.read_csv('data/processed/test.csv')

    le=LabelEncoder()
    X_train=train_df['text']
    y_train=train_df['sentiment']
    y_train_encoded=le.fit_transform(y_train)
    X_test=test_df['text']
    y_test=test_df['sentiment']
    y_test_encoded=le.transform(y_test)
    
    tokenizer=DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds=ReviewDataset(X_train, y_train, tokenizer)
    test_ds=ReviewDataset(X_test, y_test, tokenizer)
    model=DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))

    for param in model.parameters():
        param.requires_grad=False
    for param in model.distilbert.transformer.layer[5].parameters():
        param.requires_grad=True
    for param in model.pre_classifier.parameters():
        param.requires_grad=True
    for param in model.classifier.parameters():
        param.requires_grad=True
    
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')
    mlflow.log_param('trainable_params', trainable)
    mlflow.log_param('frozen_layers', 'embed+layer0-4')

    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        no_cuda=True,
        logging_steps=100,
        report_to='none'
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run('distilbert-finetuned') as run:
        mlflow.log_params({
            'model': MODEL_NAME,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'max_len': MAX_LEN,
            'num_labels': len(le.classes_)
        })
        trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
        trainer.train()
        for log in trainer.state.log_history:
            if 'eval_macro_f1' in log:
                mlflow.log_metric('eval_macro_f1', log['eval_macro_f1'], step=int(log.get('epoch', 1)))
                if 'loss' in log:
                    mlflow.log_metric('train_loss', log['loss'], step=log.get('epoch', 0))
        preds_out=trainer.predict(test_ds)
        y_pred=np.argmax(preds_out)
        clf_report=classification_report(y_test_encoded, y_pred, targets=le.classes_, output_dict=True)
        macro_f1_score=clf_report['macro avg']['f1']
        mlflow.log_metric('test_macro_f1', macro_f1_score)
        for cls in le.classes_:
            mlflow.log_metric(f'test_f1_{cls}', clf_report[cls]['f1-score'])
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        mlflow.transformers.log_model({
            'model': model,
            'tokenizer': tokenizer},
            artifact_path='distilbert-model',
            task='text-classification',
            registered_model_name='sentiment-classifier')
        logger.info(f'DistilBERT test macro F1: {macro_f1_score:.4f}')

if __name__=='__main__':
    train()
