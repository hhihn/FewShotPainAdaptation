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

## Cross-fitted block NAS protocol

The primary workflow partitions the 87 subjects into five deterministic outer
blocks. NAS for one block excludes that complete block and evaluates candidates
over three independent subject folds covering the remaining 69 or 70 subjects.
Every development subject contributes one accuracy, and Optuna maximizes mean
subject accuracy minus its standard error.

The winning trial retains the checkpoint from its strongest inner fold. For
each subject in the excluded block, that checkpoint initializes a new model, a
fresh optimizer is created, and training continues on all other 86 subjects for
the median inner best epoch. Evaluation uses only the target subject's
predefined `Test` samples. The five block searches replace 87 independent NAS
runs while keeping each target excluded from architecture selection.

## Run

The paper-scale defaults require a TensorFlow GPU:

```bash
python -m painnas cross-fitted \
  --data-dir "/content/drive/MyDrive/PainData/BioVid 2" \
  --output-dir "/content/drive/MyDrive/PainNAS/run_001" \
  --resume
```

By default, post-NAS warm-start training uses the rounded median best epoch from
the winning trial's inner folds. To force a fixed continuation length instead,
pass `--cross-fitted-continuation-epochs EPOCHS`, or set
`cross_fitted_continuation_epochs=EPOCHS` on `PainNASConfig` in the Colab
configuration cell.

With `--verbose 1` (the default), PainNAS prints intermediate loss, accuracy,
macro F1, and subject-macro validation accuracy after every NAS/continuation
epoch. It also reports epoch-level ETAs, held-out metrics after every completed
LOSO fold, and a continuation ETA for the remaining selected folds. Explicit
stage messages identify the current outer block and whether PainNAS is searching
for an architecture, has selected one, is fitting a LOSO target subject, or has
completed the block. Use `--verbose 0` to disable this progress output.

For Colab, open `painnas/colab_entrypoint.ipynb`. It checks out the repository,
installs the pinned environment, stages `BioVid.tar.gz` from Drive onto the
Colab SSD (with an extracted-Drive fallback), validates the real T0/T4 arrays,
runs/resumes block NAS and warm-started LOSO, audits target isolation, and
visualizes uncertainty-aware search and LOSO progress.

The original nested, one-time global search, and fixed-architecture LOSO remain
available as compatibility commands, but are not used by the Colab notebook:

```bash
python -m painnas search --data-dir DATA --output-dir RUN --resume
python -m painnas loso --data-dir DATA --output-dir RUN --resume
python -m painnas nested --data-dir DATA --output-dir RUN --resume
```

Cross-fitted jobs can be split across sessions without changing the manifest:

```bash
python -m painnas cross-fitted --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 1 --loso-stop-index 30
python -m painnas cross-fitted --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 31 --loso-stop-index 60
python -m painnas cross-fitted --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 61 --loso-stop-index 87
```

Use `--allow-cpu`, small epoch counts, and `--max-folds` only for debugging:

```bash
python -m painnas cross-fitted \
  --data-dir "data/BioVid 2" \
  --output-dir /tmp/painnas_smoke \
  --n-trials 1 \
  --search-max-epochs 1 \
  --outer-block-count 2 \
  --inner-fold-count 2 \
  --max-folds 1 \
  --bootstrap-samples 100 \
  --allow-cpu
```

## Outputs

- `cross_fitted_loso/blocks/block_NNN/search/study.sqlite3`: resumable study
- `cross_fitted_loso/blocks/block_NNN/search/best_warmstart.weights.h5`: winner checkpoint
- `cross_fitted_loso/blocks/block_NNN/search/best_architecture.json`: block selection
- `cross_fitted_loso/folds/fold_NNN/result.json`: continuation audit and predictions
- `cross_fitted_loso/fold_metrics.csv` and `predictions.csv`: analysis-ready tables
- `cross_fitted_loso/architecture_frequencies.csv`: selected-architecture counts
- `cross_fitted_loso/summary.json`: metrics, confusion matrix, and bootstrap CIs

Visualize a completed run as a publication-ready PNG (and optionally a vector
PDF) and print its selection and test metrics:

```bash
python -m painnas.visualize_cross_fitted_results \
  --run-dir data/cross_fitted_loso \
  --pdf
```

The selection panel reports the inner subject-macro validation accuracy used to
choose each block's architecture, including its standard error and penalized
objective. It is distinct from the held-out LOSO test accuracy and macro F1.

Resume is guarded by configuration and architecture fingerprints. If settings
change, start a new output directory.
