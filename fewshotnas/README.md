# FewShotNAS

`fewshotnas` performs a resumable, fixed 70/17 subject split architecture
search for the BioVid T0-vs-T4 CrossMod-CAN few-shot model. The 17 validation
subjects are tuning-only: zero-shot uses the learned source prototype bank;
10-shot uses repeated supports from each subject's predefined `Train` split and
the same fixed predefined `Test` query set.

Run the Colab notebook at `fewshotnas/colab_entrypoint.ipynb`, or use:

```bash
python -m fewshotnas all --data-dir DATA --output-dir RUN --resume
```

The objective gives equal weight to validation subjects and to zero-/10-shot
accuracy. The modality-specific EEGNet frontends are fixed; Optuna searches
CrossMod fusion/Transformer choices and the downstream CAN classifier. Each
candidate uses the standard sampled-support phase followed by learned-prototype
bank initialization and phase-2 fine-tuning before zero-shot validation.
