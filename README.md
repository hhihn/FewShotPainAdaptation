# FewShotPainAdaptation

Few-shot learning experiments for personalized pain assessment from physiological
signals. The repository trains episodic classifiers under leave-one-subject-out
(LOSO) evaluation, with support for EEGNet-style encoders, CrossMod feature-map
fusion, CAN/CAM attention, learned prototype memory, and BioVid Part A or
PainMonit-style NumPy datasets.

## What Is In This Repository

- `architecture/`: TensorFlow/Keras model components.
  - `eegnet_style_encoder.py`: compact EEGNet-style physiological encoder.
  - `crossmod_feature_map_encoder.py`: EDA/ECG CrossMod feature-map encoder.
  - `crossattention_module.py`: CAN/CAM query-prototype attention.
  - `learned_prototype_memory.py`: trainable class prototype-map slots.
  - `mulitmodal_proto_net.py`: multimodal prototypical network model.
- `data_loaders/`: dataset configuration, BioVid/PainMonit loading, LOSO folds,
  and episodic task sampling.
- `learner/`: training lifecycle, task batching, evaluation, checkpointing,
  adaptation, and result recording services.
- `tests/full_loso_trial.py`: command-line full LOSO training/evaluation entry point.
- `tests/quick_fewshot_trial.py`: shorter sanity-check training run.
- `scripts/`: data conversion, mock data creation, diagnostics, and plotting tools.
- `main.ipynb`: Colab-oriented notebook for BioVid Part A training.

## Installation

This project is Python-based and uses TensorFlow/Keras.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` currently pins `tensorflow==2.18.1` and includes
`tensorflow-metal` for Apple Silicon environments. On non-macOS machines you may
need to remove or skip `tensorflow-metal`.

## Data

Two dataset sources are supported through `PainDatasetConfig.dataset_source`.

### BioVid Part A

The default full LOSO script expects BioVid Part A in one of these layouts under
`--data-dir`:

```text
BioVid/PartA/Train/<MODALITY>/*_data.npy|npz
BioVid/PartA/Train/<MODALITY>/*_label.npy|npz
BioVid/PartA/Test/<MODALITY>/*_data.npy|npz
BioVid/PartA/Test/<MODALITY>/*_label.npy|npz
```

or directly:

```text
PartA/Train/...
PartA/Test/...
```

Configured modalities default to `GSR`, `ECG`, and `EMG`. When
`--encoder-backend crossmod` is used, the config narrows the input to EDA/GSR and
ECG because CrossMod is an EDA/ECG feature-map encoder.

Use the converter if you have raw BioVid Part A `.npy` files that should be
compressed to `.npz`:

```bash
python3 scripts/convert_biovid_parta_npy_to_npz.py --help
```

### PainMonit-Style NumPy Data

PainMonit-style data are flat arrays:

```text
data/X_pre.npy
data/y_heater.npy
data/subjects.npy
```

The mock-data script creates compatible placeholder arrays for local smoke tests:

```bash
python3 scripts/create_mock_pain_dataset.py
```

Then run scripts with:

```bash
--dataset-source painmonit --data-variant mock --data-dir data
```

## Running Experiments

### Quick Sanity Check

Use `quick_fewshot_trial.py` for a short single-fold run before launching LOSO.

```bash
python3 tests/quick_fewshot_trial.py \
  --data-dir data \
  --dataset-source painmonit \
  --data-variant mock \
  --updates 2 \
  --task-batch-size 2 \
  --val-tasks 2 \
  --heldout-tasks 2
```

### Full LOSO Run

The full runner trains and evaluates across held-out subjects, writes progress
CSVs, optionally saves a model summary, and emits a JSON payload.

```bash
python3 tests/full_loso_trial.py \
  --data-dir data \
  --dataset-source biovid_part_a \
  --attention-mode can \
  --encoder-backend crossmod \
  --can-support-mode learned_prototype_memory \
  --k-shot 10 \
  --q-query 10 \
  --num-epochs 1 \
  --tasks-per-epoch 20000 \
  --task-batch-size 16 \
  --val-tasks 50 \
  --heldout-eval-tasks 500 \
  --output-json outputs/full_loso/full_loso_results.json
```

For debugging, limit folds:

```bash
python3 tests/full_loso_trial.py --data-dir data --max-folds 1
```

Or select a one-based fold range:

```bash
python3 tests/full_loso_trial.py \
  --data-dir data \
  --loso-start-index 1 \
  --loso-stop-index 5
```

## Model Configuration Notes

The encoder frontend hyperparameters are unified under the `eegnet_*` knobs:

```text
eegnet_temporal_filters
eegnet_depth_multiplier
eegnet_separable_filters
eegnet_temporal_kernel_size
eegnet_separable_kernel_size
eegnet_pool_size_1
eegnet_pool_size_2
eegnet_dropout_rate
eegnet_l2_weight
```

These settings are used by both:

- `encoder_backend=eegnet`: joint multichannel EEGNet-style encoder.
- `encoder_backend=crossmod`: per-modality EEGNetStyleEncoder branches for EDA
  and ECG before CrossMod attention.

CrossMod-specific knobs now cover only attention/fusion behavior:

```text
crossmod_num_heads
crossmod_hidden_dim
crossmod_num_layers
crossmod_positional_base
crossmod_attention_dropout_rate
crossmod_ff_activation
```

`encoder_backend=crossmod` requires `attention_mode=can` and `num_sensors=2`.

## Outputs

Common output artifacts:

- `outputs/training_progress/*.csv`: per-fold training, validation, adaptation,
  and held-out event logs.
- `outputs/model_architecture/model_summary.txt`: Keras model and encoder
  summaries.
- `outputs/full_loso/full_loso_results.json`: full LOSO summary payload.
- `*_can_alignment_summary.csv`: per-fold CAN score summary when CAN is enabled.
- `*_can_sample_statistics.csv`: per-query CAN diagnostic rows when CAN is enabled.

Useful diagnostic scripts:

```bash
python3 scripts/compare_loso_progress.py --help
python3 scripts/analyze_can_alignment_performance.py --help
python3 scripts/analyze_can_sample_statistics.py --help
python3 scripts/plot_zero_shot_training_progress.py --help
```

## Testing

Run the contract and service tests with pytest:

```bash
python3 -m pytest tests -q
```

If `pytest` is not installed in your environment:

```bash
pip install pytest
```

For lightweight syntax checks:

```bash
python3 -m py_compile architecture/*.py data_loaders/*.py learner/*.py tests/*.py
```

## Reproducibility

`PainDatasetConfig` exposes:

- `seed`
- `deterministic_ops`
- `logging_verbosity`
- `validation_checkpoint_metric`
- `validation_checkpoint_mode`

Set `--deterministic-ops` on CLI runs when exact reproducibility is more important
than throughput.

## Current Caveats

- The repository uses the historical filename `architecture/mulitmodal_proto_net.py`
  (note the spelling) because existing imports depend on it.
- BioVid Part A data are not included in the repository.
- Some checked-in `data/`, `outputs/`, and notebook artifacts may be local
  experiment outputs rather than required source files.
