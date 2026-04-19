from __future__ import annotations
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

RAW_PATH='/opt/airflow/data/raw/Electronics_5.json.zip'
OUT_PATH='/opt/airflow/data/processed/reviews_clean.csv'
STAT_PATH='/opt/airflow/data/baseline/baseline_stats.json'

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
    import pandas as pd, json, zipfile, os, logging
    logger=logging.getLogger(__name__)
    logger.info('Loading raw reviews.....')
    records=[]
    with zipfile.ZipFile(RAW_PATH, 'r') as z:
        for name in z.namelist():
            with z.open(name) as f:
                for line in f:
                    try: records.append(json.loads(line.strip()))
                    except json.JSONDecodeError: continue
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
    os.makedirs('opt/airflow/data/processed', exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved {len(final_df)} reviews to {OUT_PATH}")

def compute_baseline():
    import pandas as pd, numpy as np, json, os, logging
    logger=logging.getLogger(__name__)
    df=pd.read_csv(OUT_PATH)
    lengths=df['text'].str.len()
    stats={
        'text_length_mean': float(lengths.mean()),
        'text_length_std': float(lengths.std()),
        'text_length_p95': float(np.percentile(lengths, 95)),
        'sentiment_dist': df['sentiment'].value_counts(normalize=True).to_dict(),
        'aspect_dist': df['aspect'].value_counts(normalize=True).to_dict(),
        'n_samples': len(df)
    }
    os.makedirs('opt/airflow/data/baseline', exist_ok=True)
    with open(STAT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Baseline stats saved to {STAT_PATH}")

with DAG(
    'ingest_reviews',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    
    t1=PythonOperator(task_id='ingest_and_clean', python_callable=ingest_and_clean)
    t2=PythonOperator(task_id='compute_baseline', python_callable=compute_baseline)

    t1 >> t2








            