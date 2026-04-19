from mlflow.client import MlflowClient

client=MlflowClient('http://127.0.0.1:5000')
versions=client.get_latest_versions('sentiment-classifier')
best=max(versions, key=lambda v: float(client.get_run(v.run_id).data.metrics.get('test_macro_f1', client.get_run(v.run_id).data.metrics.get('macro_f1', 0))))
client.transition_model_version_stage(name='sentiment-classifier', version=best.version, stage='Production')
print(f'Promoted version {best.version} to Production')
