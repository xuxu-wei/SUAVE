# Benchmark suite

The benchmark workflow is orchestrated by `tools/benchmark.py`. It evaluates all
hard tasks and their missingness variants using the unified model factory.

## Running the benchmark

```bash
python tools/benchmark.py --suave-epochs 20 --autogluon-time-limit 180
```

The script automatically discovers the hard tasks via
`tests.utils.benchmark_tasks.get_hard_task_configs` and generates the two
missingness variants (`*-lite` and `*-heavy`) for each task. Results are written
to `reports/baselines/candidate.json`, and the file is promoted to
`reports/baselines/current.json` if it does not yet exist.

## AutoGluon installation

`autogluon.tabular` is not part of the project dependencies. When the benchmark
script runs it attempts to install the package into the current environment on
first use. If the installation fails the suite continues, skipping the
AutoGluon baseline and recording the reason in the JSON report.

To skip the on-demand installation (for example when the environment has no
network access) pass `--skip-autogluon-install`.
