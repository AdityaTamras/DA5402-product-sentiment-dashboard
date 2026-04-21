import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR=os.path.join(BASE_DIR, 'data', 'processed')
reviews_file_path=os.path.join(PROCESSED_DIR, 'reviews_clean.csv')
train_file_path=os.path.join(PROCESSED_DIR, 'train.csv')
test_file_path=os.path.join(PROCESSED_DIR, 'test.csv')

def build_features():
    df=pd.read_csv(reviews_file_path)
    df=df[['text', 'sentiment']]
    df=df.dropna()
    df=df[df['text'].str.len()>10]
    df=df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    logger.info(f"Train : {len(train_df)} rows  | Test : {len(test_df)} rows")

if __name__=="__main__":
    build_features()