"""Post-retrain diagnostic: Check predictions from fresh models."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.ml.pipeline.processor import DataProcessor
from app.blueprints.dashboard.routes import run_inference
import numpy as np
import pandas as pd

processor = DataProcessor()
train_df, test_df = processor.prepare_training_data()
processor.close()

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"Train VOTE_SHARE: mean={train_df['VOTE_SHARE'].mean():.2f}, max={train_df['VOTE_SHARE'].max():.2f}")
print(f"Test  VOTE_SHARE: mean={test_df['VOTE_SHARE'].mean():.2f}, max={test_df['VOTE_SHARE'].max():.2f}")

# Test each model individually
for model_name in ['rf', 'lgbm', 'catboost', 'xgboost', 'nn', 'elasticnet', 'Ensemble']:
    try:
        result_df, label = run_inference(test_df, model_name)
        if result_df is not None:
            pred = result_df['PREDICTED']
            actual = result_df['VOTE_SHARE']
            diff = (pred - actual).abs()
            print(f"\n{model_name.upper():12s} | pred mean={pred.mean():8.3f} | actual mean={actual.mean():8.3f} | MAE={diff.mean():.3f} | pred_zeros={(pred < 0.01).sum()}/{len(pred)}")
        else:
            print(f"\n{model_name.upper():12s} | FAILED (returned None)")
    except Exception as e:
        print(f"\n{model_name.upper():12s} | ERROR: {e}")

# Show aggregated chart comparison for Ensemble
print(f"\n{'='*60}")
print(f"AGGREGATED COMPARISON (Ensemble):")
result_df, _ = run_inference(test_df, "Ensemble")
if result_df is not None:
    party_agg = result_df.groupby('SARASO_PAVADINIMAS').apply(lambda x: pd.Series({
        'ACTUAL': (x['BALSU_VISO'].sum() / max(1, x['VISO_DALYVAVO'].sum())) * 100,
        'PREDICTED': ((x['PREDICTED'] * x['VISO_DALYVAVO']).sum() / max(1, x['VISO_DALYVAVO'].sum()))
    }), include_groups=False).reset_index()
    top = party_agg.sort_values('ACTUAL', ascending=False).head(8)
    top['DELTA'] = top['PREDICTED'] - top['ACTUAL']
    for _, row in top.iterrows():
        name = row['SARASO_PAVADINIMAS'][:40]
        print(f"  {name:42s} Actual={row['ACTUAL']:6.2f}%  Pred={row['PREDICTED']:6.2f}%  Δ={row['DELTA']:+.2f}")
