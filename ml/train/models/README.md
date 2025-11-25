# Trained Models Directory

This directory contains trained ML models for the AURA K8s system.

## Model Files

### Included in Git (Smaller Models)
- `xgboost_model.joblib` (1.2MB)
- `random_forest_model.joblib` (3.3MB)
- `lightgbm_model.joblib` (3.8MB)
- `gradient_boosting_model.joblib` (5.7MB)
- `catboost_model.joblib` (49MB)
- `isolation_forest_model.joblib` (2.7MB)
- `scaler.joblib` (4KB)
- `label_encoder.joblib` (1.8KB)
- `feature_selector.joblib` (3.4KB)

### Excluded from Git (Too Large)
- `ensemble_model.joblib` (133MB) - **Can be reconstructed from individual models**

### Metadata Files (Always in Git)
- `feature_names.json`
- `anomaly_types.json`
- `training_metadata.json`
- `selected_feature_names.json`
- `test_results.json`

## Auto-Reconstruction

The `ensemble_model.joblib` (133MB) is excluded from git because it exceeds GitHub's 100MB limit. However:

1. **The ensemble can be reconstructed** from individual models automatically
2. **The ML service** will create the ensemble on startup if needed
3. **No retraining required** - individual models are sufficient

## Model Loading

The system automatically:
- Loads individual models if they exist
- Reconstructs ensemble from individual models if needed
- Skips training if models are present
- Only trains if models are missing

## Force Retraining

To force retraining (if you want to update models):
```bash
rm -rf ml/train/models/*.joblib
python ml/train/beast_train.py
```

Or set environment variable:
```bash
FORCE_RETRAIN=true python ml/train/beast_train.py
```

## Best Practice

- ✅ Individual models preserved in git (no retraining needed)
- ✅ Ensemble reconstructed automatically (no manual steps)
- ✅ Fast startup (models load instantly)
- ✅ Respects GitHub file size limits

