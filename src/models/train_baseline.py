import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR=os.path.join(BASE_DIR, 'data', 'processed')
train_file_path=os.path.join(PROCESSED_DIR, 'train.csv')
test_file_path=os.path.join(PROCESSED_DIR, 'test.csv')
MODELS_DIR=os.path.join(BASE_DIR, 'models')
model_path=os.path.join(MODELS_DIR, 'tfidf_lr_model.pkl')
le_path=os.path.join(MODELS_DIR, 'label_encoder.pkl')
MLFLOW_URI='http://127.0.0.1:5000'
EXPERIMENT='sentiment-classification'

def train():
    train_df=pd.read_csv(train_file_path)
    test_df=pd.read_csv(test_file_path)

    X_train=train_df['text']
    y_train=train_df['sentiment']
    X_test=test_df['text']
    y_test=test_df['sentiment']

    le=LabelEncoder()
    y_train_encoded=le.fit_transform(y_train)
    y_test_encoded=le.transform(y_test)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name='tfidf-lr-baseline') as run:
        params={
            'max_features': 50000,
            'ngram_range': (1, 2),
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        mlflow.log_params(params)
        pipeline=Pipeline([
            ('tfidf', TfidfVectorizer(max_features=params['max_features'], ngram_range=params['ngram_range'])),
            ('clf', LogisticRegression(C=params['C'], max_iter=params['max_iter'], class_weight=params['class_weight'], solver='lbfgs'))
            ])
        pipeline.fit(X_train, y_train_encoded)
        preds=pipeline.predict(X_test)
        macro_f1=f1_score(y_test_encoded, preds, average='macro')
        clf_report=classification_report(y_test_encoded, preds, target_names=le.classes_, output_dict=True)
        mlflow.log_metric('macro_f1_score', macro_f1)
        for cls in le.classes_:
            mlflow.log_metric(f'f1_score_{cls}', clf_report[cls]['f1-score'])
            mlflow.log_metric(f'precision_{cls}', clf_report[cls]['precision'])
            mlflow.log_metric(f'recall_{cls}', clf_report[cls]['recall'])
        mlflow.sklearn.log_model(pipeline, 'model', registered_model_name='sentiment-classifier')
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)

        logger.info(f'Baseline Macro-F1 : {macro_f1:.4f}')
        return run.info.run_id
    
if __name__=='__main__':
    train()
    
        