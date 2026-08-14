# PainNAS

`painnas` is a standalone supervised baseline for automated pain assessment on
BioVid Part A. It deliberately does not import or use the repository's episodic
samplers, prototypes, support/query construction, or few-shot adaptation.

The input is early-fused in the order GSR/EDA, ECG, EMG and transformed from the
repository layout `[batch, 1152, 3]` to `[batch, 3, 1152, 1]`. Only `T0` and `T4`
are retained. The exact supplied Table 2 CNN is enqueued as Optuna trial 0; later
trials may change convolutional depth, width, temporal kernel size, dense depth,
dense width, and Adam learning rate. ELU, softmax, `(1, 2)` pooling, and dropout
`0.25` remain fixed.

## Protocol warning

The selected protocol performs architecture search once using all BioVid
subject identities and then reuses that architecture in 87 fresh LOSO trainings.
This is computationally practical but is not an unbiased nested-LOSO estimate.
Every output manifest records this limitation.

By default, the one-time search deterministically assigns 70 subjects to search
training and 17 subjects to search validation. Search training uses those 70
subjects' predefined `Train` samples; validation uses the 17 held-out subjects'
predefined `Test` samples. No NAS weights are carried into LOSO.

Within each LOSO fold, the target subject is excluded from training,
source-validation, and source-only normalization. The other 86 subjects'
predefined `Train` samples train the fold, their `Test` samples provide early
stopping, and only the target subject's `Test` samples provide fold metrics. The
model and optimizer are newly constructed for every fold.

## Run

The paper-scale defaults require a TensorFlow GPU:

```bash
python -m painnas all \
  --data-dir "/content/drive/MyDrive/PainData/BioVid 2" \
  --output-dir "/content/drive/MyDrive/PainNAS/run_001" \
  --resume
```

For Colab, open `painnas/colab_entrypoint.ipynb`. It checks out the repository,
installs the pinned environment, stages `BioVid.tar.gz` from Drive onto the
Colab SSD (with an extracted-Drive fallback), validates the real T0/T4 arrays,
runs/resumes NAS, visualizes trial progress, and optionally continues with LOSO.
Only the durable search and fold artifacts are stored in Drive.

Stages may be run separately:

```bash
python -m painnas search --data-dir DATA --output-dir RUN --resume
python -m painnas loso --data-dir DATA --output-dir RUN --resume
```

Long LOSO jobs can be split across sessions without changing the manifest:

```bash
python -m painnas loso --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 1 --loso-stop-index 30
python -m painnas loso --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 31 --loso-stop-index 60
python -m painnas loso --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 61 --loso-stop-index 87
```

Use `--allow-cpu`, small epoch counts, and `--max-folds` only for debugging:

```bash
python -m painnas all \
  --data-dir "data/BioVid 2" \
  --output-dir /tmp/painnas_smoke \
  --n-trials 1 \
  --search-max-epochs 1 \
  --loso-max-epochs 1 \
  --search-validation-subjects 2 \
  --max-folds 1 \
  --bootstrap-samples 100 \
  --allow-cpu
```

## Outputs

- `search/study.sqlite3`: resumable Optuna study
- `search/trials.csv`: trial parameters and outcomes
- `search/best_architecture.json`: selected architecture only, without weights
- `loso/folds/fold_NNN.json`: atomic fold histories, metrics, and predictions
- `loso/fold_metrics.csv` and `loso/predictions.csv`: analysis-ready tables
- `loso/summary.json`: aggregate metrics, confusion matrix, and bootstrap CIs

Resume is guarded by configuration and architecture fingerprints. If settings
change, start a new output directory.
