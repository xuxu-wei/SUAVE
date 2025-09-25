# Coding Agent Behavior for SUAVE

## Mission
Deliver a user-friendly Python package for HI-VAE-based tabular modeling (generation + classification + calibration).
**No separate global config**: all hyperparameters live in class/method signatures. If a config is unavoidable, it must be
data-dependent and stored in the **same directory as the user's data** (e.g., schema.json).

## Golden Rules
1) **Minimal viable first**: implement the smallest feature set that runs end-to-end, then iterate.
2) **API stability > feature breadth**: keep `SUAVE` methods stable (`fit/predict/predict_proba/calibrate/encode/sample/save/load`).
3) **No hidden state**: every training- or data-dependent default is explicit in arguments or saved alongside the model.
4) **Local-only config**: if a schema file is used, write/read it ONLY from the data directory.
5) **No auto type inference (v1)**: require a user-provided `schema` dict. Provide clear error messages if incompatible.
6) **Validation split is internal to fit()**: user supplies train/test/external; `fit()` creates a val split.
7) **Numerical stability**: use softplus for variances/rates; clamp eps=1e-6; mask-aware losses for missing.
8) **Small steps, small diffs**: each request updates as few files as possible; include docstrings and type hints.
9) **Tests & style**: add/maintain pytest for each new module; keep `black` and `ruff` clean.

## Architecture (must follow)
- `SUAVE` class: constructor args expose all hyperparameters; methods must be documented with shapes & types.
- `modules/encoder.py`: MLP encoder; `modules/decoder.py`: heads for real/cat/pos/count/ordinal.
- `modules/distributions.py`: torch.distributions wrappers + NLL.
- `modules/losses.py`: ELBO (mask-aware), KL warmup; CE/Focal.
- `modules/heads.py`: classification heads; supports class_weight/focal.
- `modules/calibrate.py`: temperature scaling.
- `evaluate.py`: metrics (AUROC/AUPRC/Brier/ECE), TSTR/TRTR, MIA baseline (simple).
- `data.py`: normalization/denormalization; missing masks; internal val split.
- `sampling.py`: unconditional/conditional sampling.

## Training Schedule (when implement)
- Warm-start: train ELBO only, KL anneal 0→β.
- Head: freeze decoder, train classifier on z (class-weight or focal optional).
- Light Joint Fine-tune: unfreeze with small lr (decoder lr smaller). Early stop on Val NLL + Brier/ECE.

## Acceptance Criteria
- `examples/sepsis_minimal.py` runs end-to-end using only a manual schema dict; produces metrics and a reliability reports.
- `predict()` returns predicted label.
- `predict_proba()` returns calibrated probabilities after `calibrate()`.
- `sample(n, y=...)` works when `conditional=True`.
- `pytest -q` passes; `black . && ruff .` passes.

## Defaults & Error Handling
- Defaults aim for “works out-of-the-box”: latent_dim=32, beta=1.5, hidden=(256,128), dropout=0.1, lr=1e-3.
- Clear `ValueError` when schema types mismatch implemented heads; message must tell how to fix.
- If CUDA not available, fall back to CPU with a warning.

## Documentation
- Each public method must include an example code snippet.
- Keep examples minimal and data-directory-centric (schema in the same folder as the CSV).
- 当 `examples/` 下与研究流程相关的脚本更新时，需要同步审阅并更新 `docs/research_protocol.md`，确保协议步骤与脚本实现一致。