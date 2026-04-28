# DA5402-product-sentiment-dashboard
DA5402 MLOps Lab End-to-end project : Product Sentiment Dashboard

E-commerce businesses accumulate thousands of customer reviews continuously across products, yet the teams responsible for acting on that feedback - product managers, supply chain leads, customer service heads, have no scalable way to process it. Current practice relies on manual spot-checking, lagging aggregate star ratings, or periodic analyst reports. The result is that a genuine product quality regression, a fulfilment partner failure, or a pricing perception problem can persist for weeks while reviews accumulate and ratings decay, by which point the damage to search ranking, conversion rate, and repeat purchase behaviour has already occurred.

This project builds an automated, aspect-based sentiment analysis pipeline that ingests product reviews in near real-time, classifies sentiment at the aspect level (price, quality, delivery, customer service), and surfaces the results on a live operational dashboard, compressing the feedback loop from weeks to hours and making the signal specific enough to act on immediately.

## Project Structure

```
product-sentiment-dashboard/
│
├── .dvc/                          # DVC internal config and cache pointers
├── airflow/                       # Airflow DAGs and pipeline orchestration
├── data/                          # Raw, interim, and processed datasets (tracked by DVC)
├── docs/                          # Project documentation and references
├── frontend/                      # React application (sentiment dashboard UI)
├── monitoring/                    # Prometheus and Grafana config files
├── models/                        # Trained model artifacts and serialized files
├── src/                           # Core source code (training, inference, API)
├── tests/                       
│
├── .dockerignore                 
├── .dvcignore                    
├── .gitattributes                
├── .gitignore                    
├── Dockerfile.backend             # Docker image definition for FastAPI backend
├── README.md                     
├── docker-compose.yml             # Multi-container setup (backend, frontend, monitoring)
├── dvc.lock                       # Locked DVC pipeline stages and file hashes
├── dvc.yaml                       # DVC pipeline stage definitions
└── requirements.txt  
```

