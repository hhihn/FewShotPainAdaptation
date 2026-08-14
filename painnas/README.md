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

## Nested LOSO protocol

Every outer fold holds out one subject completely. The remaining 86 subjects
are deterministically divided into 69 inner-training and 17 inner-validation
subjects. NAS candidates train on the 69 subjects' predefined `Train` samples
and are selected by macro-F1 on the 17 subjects' predefined `Test` samples.

The winning candidate's weights are discarded. A fresh model and optimizer are
created, source normalization is recomputed from all 86 subjects' `Train`
samples, and the selected architecture is fitted for its inner-selected best
epoch. Evaluation uses only the outer target subject's `Test` samples; all of
that subject's `Train` samples remain unused. The default nested budget is 10
trials of at most 20 epochs per outer fold.

## Run

The paper-scale defaults require a TensorFlow GPU:

```bash
python -m painnas nested \
  --data-dir "/content/drive/MyDrive/PainData/BioVid 2" \
  --output-dir "/content/drive/MyDrive/PainNAS/run_001" \
  --resume
```

For Colab, open `painnas/colab_entrypoint.ipynb`. It checks out the repository,
installs the pinned environment, stages `BioVid.tar.gz` from Drive onto the
Colab SSD (with an extracted-Drive fallback), validates the real T0/T4 arrays,
runs/resumes per-fold NAS and refitting, audits target isolation, and visualizes
search and LOSO progress. Only durable search and fold artifacts are stored in
Drive.

The original one-time global search and fixed-architecture LOSO remain available
as explicitly exploratory compatibility commands:

```bash
python -m painnas search --data-dir DATA --output-dir RUN --resume
python -m painnas loso --data-dir DATA --output-dir RUN --resume
```

Nested jobs can be split across sessions without changing the manifest:

```bash
python -m painnas nested --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 1 --loso-stop-index 30
python -m painnas nested --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 31 --loso-stop-index 60
python -m painnas nested --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 61 --loso-stop-index 87
```

Use `--allow-cpu`, small epoch counts, and `--max-folds` only for debugging:

```bash
python -m painnas nested \
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

- `nested_loso/folds/fold_NNN/search/study.sqlite3`: resumable fold study
- `nested_loso/folds/fold_NNN/search/trials.csv`: fold trial outcomes
- `nested_loso/folds/fold_NNN/search/best_architecture.json`: fold selection
- `nested_loso/folds/fold_NNN/result.json`: refit history, audit, and predictions
- `nested_loso/fold_metrics.csv` and `predictions.csv`: analysis-ready tables
- `nested_loso/architecture_frequencies.csv`: selected-architecture counts
- `nested_loso/summary.json`: metrics, confusion matrix, and bootstrap CIs

Resume is guarded by configuration and architecture fingerprints. If settings
change, start a new output directory.
