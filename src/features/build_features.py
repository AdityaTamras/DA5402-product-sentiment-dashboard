import pandas as pd
from sklearn.metrics import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def build_features():
    df=pd.read_csv('data/processed/reviews_clean.csv')
    df=df[['text', 'sentiment']]
    df=df.dropna()
    df=df[df['text'].str.len()>10]
    df=df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    logger.info(f"Train : {len(train_df)} rows  | Test : {len(test_df)} rows")

if __name__=="__main__":
    build_features()