import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
import mlflow
import mlflow.transformers
import time
import logging
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR=os.path.join(BASE_DIR, 'data', 'processed')
train_file_path=os.path.join(PROCESSED_DIR, 'train.csv')
test_file_path=os.path.join(PROCESSED_DIR, 'test.csv')
MODELS_DIR=os.path.join(BASE_DIR, 'models', 'distilbert_finetuned')
MLFLOW_URI='http://127.0.0.1:5000'
EXPERIMENT='sentiment-classification'
MODEL_NAME=os.path.join(BASE_DIR, 'models', 'distilbert_base')
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
    train_df=pd.read_csv(train_file_path)
    test_df=pd.read_csv(test_file_path)

    le=LabelEncoder()
    X_train=train_df['text']
    y_train=train_df['sentiment']
    y_train_encoded=le.fit_transform(y_train)
    X_test=test_df['text']
    y_test=test_df['sentiment']
    y_test_encoded=le.transform(y_test)
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    tokenizer=DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds=ReviewDataset(X_train, y_train_encoded, tokenizer)
    test_ds=ReviewDataset(X_test, y_test_encoded, tokenizer)
    model=DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_), ignore_mismatched_sizes=True)

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

    args=TrainingArguments(
        output_dir=MODELS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        use_cpu=True,
        logging_steps=100,
        report_to='none'
    )

    mlflow.end_run()
    with mlflow.start_run(run_name='distilbert-finetuned') as run:
        mlflow.log_params({
            'model': MODEL_NAME,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'max_len': MAX_LEN,
            'num_labels': len(le.classes_),
            'trainable_params': trainable,
            'frozen_layers': "embed+layer0-4"
        })

        trainer=Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
        trainer.train()
        for log in trainer.state.log_history:
            if 'eval_macro_f1' in log:
                mlflow.log_metric('eval_macro_f1', log['eval_macro_f1'], step=int(log.get('epoch', 1)))
                if 'loss' in log:
                    mlflow.log_metric('train_loss', log['loss'], step=log.get('epoch', 0))
        preds_out=trainer.predict(test_ds)
        y_pred=np.argmax(preds_out.predictions, axis=1)
        clf_report=classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
        macro_f1_score=clf_report['macro avg']['f1-score']
        mlflow.log_metric('test_macro_f1', macro_f1_score)
        for cls in le.classes_:
            mlflow.log_metric(f'test_f1_{cls}', clf_report[cls]['f1-score'])
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        trainer.save_model(MODELS_DIR)
        tokenizer.save_pretrained(MODELS_DIR)

        mlflow.transformers.log_model({
            'model': model,
            'tokenizer': tokenizer},
            artifact_path='distilbert-model',
            task='text-classification',
            registered_model_name='sentiment-classifier')
        logger.info(f'DistilBERT test macro F1: {macro_f1_score:.4f}')

if __name__=='__main__':
    train()
