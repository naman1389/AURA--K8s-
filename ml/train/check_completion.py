#!/usr/bin/env python3
"""
Check if training completed successfully and all required files are created
"""

import json
from pathlib import Path
from datetime import datetime

def check_training_completion():
    """Check if all required files are present after training"""
    models_dir = Path("models")
    
    print("="*70)
    print("📊 TRAINING COMPLETION CHECK")
    print("="*70)
    print()
    
    # Required files
    required_files = {
        'Models': [
            'xgboost_model.joblib',
            'lightgbm_model.joblib',
            'catboost_model.joblib',
            'random_forest_model.joblib',
            'isolation_forest_model.joblib',
            'ensemble_model.joblib'
        ],
        'Artifacts': [
            'scaler.joblib',
            'label_encoder.joblib',
            'feature_selector.joblib'
        ],
        'Metadata': [
            'feature_names.json',
            'selected_feature_names.json',
            'anomaly_types.json',
            'training_metadata.json'
        ]
    }
    
    all_found = []
    all_missing = []
    
    print("✅ CHECKING REQUIRED FILES:")
    print("-"*70)
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for filename in files:
            filepath = models_dir / filename
            if filepath.exists():
                # Check if recently created (last 2 hours)
                mtime = filepath.stat().st_mtime
                age_minutes = (datetime.now().timestamp() - mtime) / 60
                
                if age_minutes < 120:
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {filename:35s} ({size_mb:.2f} MB, {age_minutes:.1f} min ago)")
                    all_found.append(filename)
                else:
                    print(f"   ⚠️  {filename:35s} (OLD - {age_minutes/60:.1f} hours ago)")
                    all_missing.append(filename)
            else:
                print(f"   ❌ {filename:35s} (MISSING)")
                all_missing.append(filename)
    
    print()
    print("="*70)
    print("📊 SUMMARY:")
    print("-"*70)
    
    total_required = sum(len(files) for files in required_files.values())
    total_found = len(all_found)
    
    print(f"Files Found: {total_found}/{total_required} ({total_found/total_required*100:.1f}%)")
    print(f"Files Missing: {len(all_missing)}/{total_required}")
    
    if all_missing:
        print()
        print("❌ MISSING FILES:")
        for filename in all_missing:
            print(f"   • {filename}")
    
    print()
    
    # Check training metadata if exists
    metadata_file = models_dir / 'training_metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                meta = json.load(f)
            
            print("📋 TRAINING METADATA:")
            print("-"*70)
            print(f"   Version: {meta.get('version', 'unknown')}")
            print(f"   Training Date: {meta.get('training_date', 'unknown')}")
            print(f"   Features: {meta.get('num_features', 'unknown')}")
            print(f"   Selected Features: {meta.get('num_selected_features', 'unknown')}")
            print(f"   Anomaly Types: {meta.get('num_anomaly_types', 'unknown')}")
            
            if 'model_performance' in meta:
                print()
                print("   Model Performance:")
                for model_name, perf in meta['model_performance'].items():
                    acc = perf.get('accuracy', 0)
                    f1 = perf.get('f1_score', 0)
                    print(f"      • {model_name:20s}: Accuracy={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"   ⚠️  Error reading metadata: {e}")
    
    print()
    print("="*70)
    
    if len(all_missing) == 0 and total_found == total_required:
        print("✅ ALL FILES CREATED SUCCESSFULLY!")
        print("✅ Training completed successfully!")
        return True
    elif total_found >= total_required * 0.8:
        print("⚠️  MOST FILES CREATED (80%+)")
        print("⚠️  Training may still be in progress or some files failed to save")
        return False
    else:
        print("❌ INCOMPLETE - Training may still be in progress")
        return False

if __name__ == "__main__":
    check_training_completion()

