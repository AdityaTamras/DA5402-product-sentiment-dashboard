import pandas as pd
import numpy as np
import json
import zipfile
import os
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s -- %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH=os.path.join(BASE_DIR, 'data', 'raw', 'Electronics_5.json.zip')
OUT_PATH=os.path.join(BASE_DIR, 'data', 'processed', 'reviews_clean.csv')
STAT_PATH=os.path.join(BASE_DIR, 'data', 'baseline', 'baseline_stats.json')


ASPECT_KEYWORDS={
    'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'worth', 'value'],
    'quality': ['quality', 'material', 'build', 'durable', 'broke', 'excellent', 'poor'],
    'delivery': ['shipping', 'delivery', 'arrived', 'dispatch', 'late', 'fast', 'slow'],
    'service': ['service', 'support', 'return', 'refund', 'customer', 'helpful', 'rude']
}

def assign_aspect(text):
    text_lower=str(text).lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
         if any(k in text_lower for k in keywords):
              return aspect
    return 'general'

def assign_sentiment(rating):
    if int(rating)>=4:
        return 'positive'
    elif int(rating)==3:
        return 'neutral'
    else:
        return 'negative'

def ingest_and_clean():
    logger.info('Step 1/2 - Loading raw reviews.....')
    records=[]
    append_record=records.append
    json_loads=json.loads
    with zipfile.ZipFile(RAW_PATH, 'r') as z:
        file_names=[name for name in z.namelist() if not name.endswith("/")]
        for name in z.namelist():
            with z.open(name) as f:
                for line in f:
                    try: append_record(json_loads(line))
                    except json.JSONDecodeError: continue
    logger.info(f'Loaded {len(records)} raw records')
    df=pd.DataFrame(records)
    df=df[['reviewText', 'overall', 'summary']]
    df=df.dropna(subset=['reviewText', 'overall'])
    df=df[df['reviewText'].str.len()>20]
    df=df.drop_duplicates(subset='reviewText')
    df['overall']=pd.to_numeric(df['overall'], errors='coerce').astype(int)
    df['sentiment']=df['overall'].apply(assign_sentiment)
    df['aspect']=df['reviewText'].apply(assign_aspect)
    df['text']=df['reviewText'].str.strip()
    final_df=df.drop(columns=['reviewText', 'summary']).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved {len(final_df)} reviews to {OUT_PATH}")
    return final_df

def compute_baseline(df):
    logger.info('Step 2/2 - Computing baseline statistics......')
    lengths=df['text'].str.len()
    stats={
        'text_length_mean': float(lengths.mean()),
        'text_length_std': float(lengths.std()),
        'text_length_p95': float(np.percentile(lengths, 95)),
        'sentiment_dist': df['sentiment'].value_counts(normalize=True).to_dict(),
        'aspect_dist': df['aspect'].value_counts(normalize=True).to_dict(),
        'n_samples': len(df)
    }
    os.makedirs(os.path.dirname(STAT_PATH), exist_ok=True)
    with open(STAT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Baseline stats saved to {STAT_PATH}")

if __name__=='__main__':
    df=ingest_and_clean()
    compute_baseline(df)
    logger.info('=== Pipeline complete ===')
