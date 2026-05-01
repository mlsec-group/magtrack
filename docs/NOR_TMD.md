## NOR-TMD reimplementation

This part reimplements the NOR-TMD evaluation pipeline as a 4-stage CLI workflow:
1. parse raw CSV from zip into DuckDB
2. preprocess sensor streams into rolling-window statistics
3. pivot sensor ids to model-ready feature columns
4. train/evaluate an XGBoost classifier

### Dataset

Download the NOR-TMD dataset from Kaggle:
https://www.kaggle.com/datasets/scholarone/nor-tmd

You need the dataset zip file path for the CLI `--zip-path` argument.

### Reproduce results (zipfile argument + 10 runs)

Run the full pipeline end-to-end with 10 training runs:

```bash
evaluate-nor-tmd --zip-path path/to/nor_tmd.zip --runs 10
```

Notes:
- `--zip-path` is required.
- `--runs 10` performs 10 train/eval runs and prints a mean+-std summary table.

### Stage-by-stage usage (optional)

If you want to run each stage separately:

```bash
evaluate-nor-tmd parse --zip-path path/to/nor_tmd.zip
evaluate-nor-tmd preprocess
evaluate-nor-tmd pivot
evaluate-nor-tmd train --runs 10
```

### Output artifacts (default paths)

All outputs are written under `data/nor_tmd/` by default:
- `nor_tmd_complete.db` (DuckDB from parse)
- `segmented_df.parquet` (preprocessed windows)
- `data_android_centered.parquet` (pivoted Android dataset used for train)
- `data_ios_centered.parquet` (pivoted iOS dataset)
