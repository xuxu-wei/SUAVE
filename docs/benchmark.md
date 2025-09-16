# Hard Benchmark Runner

The benchmark runner evaluates SUAVE and baseline models on the synthetic hard
tasks as well as the new missingness variants. Results are stored in
`reports/baselines/candidate.json` and merged into
`reports/baselines/current.json` for regression checks.

## Usage

```bash
python tools/benchmark.py \
    --epochs 20 \
    --latent-dim 8 \
    --autogluon-time-limit 120
```

Optional arguments:

- `--max-train-samples`: limit the number of training rows for each task.
- `--output`: override the location of `candidate.json`.
- `--current`: override the location of `current.json`.

During the first run the script attempts to import
`autogluon.tabular.TabularPredictor`. If the package is missing the script
installs it in the active environment and proceeds with evaluation. The library
is only required for this benchmark script; the core SUAVE package does not
list AutoGluon as an installation dependency.

The console output prints progress for each task. After completion, compare the
new candidate against the baseline with:

```bash
python tools/compare_baselines.py
```
