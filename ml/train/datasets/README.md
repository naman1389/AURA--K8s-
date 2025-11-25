# Datasets Directory

This directory contains training datasets for the AURA K8s ML models.

## Best Practice: Auto-Download on Startup

Datasets are **not stored in git** (too large - 397MB+). Instead, they are automatically downloaded when needed.

### Automatic Download

The system will automatically download datasets when:
- Running `ml/train/beast_train.py` for the first time
- Using `ml/train/dataset_downloader.py` directly
- Starting the system with `./start.sh` (if training is needed)

### Manual Download

To manually download datasets:

```bash
cd ml/train
python dataset_downloader.py
```

### Available Datasets

1. **synthetic_k8s** - Synthetic Kubernetes metrics (small, included)
2. **yahoo_s5** - Yahoo S5 Time Series Anomaly Detection
3. **numenta** - Numenta Anomaly Benchmark (NAB)
4. **kdd_cup_99** - KDD Cup 1999 Network Intrusion (optional)

### Directory Structure

```
datasets/
├── synthetic_k8s/          # Small synthetic data (may be in git)
├── yahoo_s5/               # Auto-downloaded
├── numenta/                # Auto-downloaded (308MB)
└── kdd_cup_99/             # Auto-downloaded (optional)
```

### Why Not in Git?

- **Size**: Total datasets are 397MB+ (too large for git)
- **Best Practice**: Large data files should be downloaded, not versioned
- **Flexibility**: Users can choose which datasets to download
- **Performance**: Faster git operations without large files

### Git LFS Alternative

For production deployments that need datasets in version control, consider using Git LFS:

```bash
git lfs install
git lfs track "ml/train/datasets/**/*.csv"
git lfs track "ml/train/datasets/**/*.json"
git add .gitattributes
```

However, the current approach (auto-download) is recommended for most use cases.

