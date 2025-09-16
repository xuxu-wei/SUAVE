"""Benchmark SUAVE on the hard benchmark for various beta/InfoVAE settings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suave.api import InfoVAEConfig
from suave.sklearn import SuaveClassifier
from tests.utils.task_registry import load_task


@dataclass(frozen=True)
class ExperimentSetting:
    """Container describing one configuration under evaluation."""

    label: str
    beta: float = 1.0
    info_config: InfoVAEConfig | None = None

    def to_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {"beta": self.beta}
        if self.info_config is not None:
            kwargs["info_config"] = self.info_config
        return kwargs

    def describe(self) -> Dict[str, object]:
        data: Dict[str, object] = {"label": self.label, "beta": self.beta}
        data["info_config"] = asdict(self.info_config) if self.info_config is not None else None
        return data


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_split(
    seed: int, max_attempts: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], int]:
    task = load_task("hard-core")
    X, Y, task_classes = task.features, task.targets, task.task_classes
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        random_state = int(rng.integers(0, 1_000_000))
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )
        if all(np.unique(y_train[:, idx]).size == cls for idx, cls in enumerate(task_classes)):
            return X_train, X_test, y_train, y_test, task_classes, random_state
    raise RuntimeError("Failed to find a split covering all classes")


def _preprocess(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer()
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))
    return X_train_proc, X_test_proc


def _compute_task_metrics(
    y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if num_classes > 2:
        classes = np.arange(num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
        metrics["auroc_micro"] = float(roc_auc_score(y_true_bin, proba, average="micro"))
        metrics["auprc_macro"] = float(average_precision_score(y_true_bin, proba, average="macro"))
        metrics["auprc_micro"] = float(average_precision_score(y_true_bin, proba, average="micro"))
        metrics["acc_top1"] = float(accuracy_score(y_true, pred))
        metrics["f1_macro"] = float(f1_score(y_true, pred, average="macro"))
    else:
        metrics["auroc_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
        metrics["auroc_micro"] = metrics["auroc_macro"]
        metrics["auprc_macro"] = float(average_precision_score(y_true, proba[:, 1]))
        metrics["auprc_micro"] = metrics["auprc_macro"]
        metrics["acc_top1"] = float(accuracy_score(y_true, pred))
        metrics["f1_macro"] = float(f1_score(y_true, pred, average="macro"))
    return metrics


def _loss_tuple_to_dict(values: Tuple[float, float, float, float, float, float]) -> Dict[str, float]:
    keys = ["total", "reconstruction", "kl", "classification", "info_penalty", "info_weight"]
    return {key: float(val) for key, val in zip(keys, values)}


def run_experiments(
    settings: Iterable[ExperimentSetting],
    *,
    split_seed: int,
    epochs: int,
    base_model_seed: int,
    output: Path,
    max_train_samples: int | None,
) -> None:
    X_train, X_test, y_train, y_test, task_classes, split_random_state = _prepare_split(split_seed)
    X_train_proc, X_test_proc = _preprocess(X_train, X_test)

    if max_train_samples is not None and max_train_samples < X_train_proc.shape[0]:
        sample_rng = np.random.default_rng(split_seed + 1337)
        idx = sample_rng.choice(X_train_proc.shape[0], max_train_samples, replace=False)
        X_train_proc = X_train_proc[idx]
        y_train = y_train[idx]

    settings_list = list(settings)
    results = []
    table_rows = []

    for idx, setting in enumerate(settings_list, start=1):
        print(f"[{idx}/{len(settings_list)}] running {setting.label}")
        _set_seed(base_model_seed + idx)
        model = SuaveClassifier(
            input_dim=X_train_proc.shape[1],
            task_classes=task_classes,
            latent_dim=8,
            **setting.to_kwargs(),
        )
        model.fit(X_train_proc, y_train, epochs=epochs)
        aucs = model.score(X_test_proc, y_test)
        probas = model.predict_proba(X_test_proc)
        preds = model.predict(X_test_proc)
        metrics = {}
        losses = {}
        for task_idx, num_classes in enumerate(task_classes):
            task_name = f"y{task_idx + 1}"
            y_true = y_test[:, task_idx]
            metrics[task_name] = _compute_task_metrics(
                y_true, probas[task_idx], preds[task_idx], num_classes
            )
            losses[task_name] = _loss_tuple_to_dict(
                model.models[task_idx].eval_loss(X_test_proc, y_true)
            )
        results.append(
            {
                "setting": setting.describe(),
                "split_random_state": split_random_state,
                "epochs": epochs,
                "score": {f"y{t + 1}": float(val) for t, val in enumerate(aucs)},
                "metrics": metrics,
                "losses": losses,
            }
        )
        table_rows.append(
            {
                "setting": setting.label,
                **{f"y{t + 1}_auroc": float(val) for t, val in enumerate(aucs)},
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_seed": split_seed,
        "split_random_state": split_random_state,
        "epochs": epochs,
        "train_samples": int(X_train_proc.shape[0]),
        "results": results,
    }
    with output.open("w") as f:
        json.dump(payload, f, indent=2)

    md_path = output.with_suffix(".md")
    lines = ["# Hard benchmark: beta/InfoVAE sweep", ""]
    lines.append(f"- Split seed generator: {split_seed}")
    lines.append(f"- Train/test split random_state: {split_random_state}")
    lines.append(f"- Epochs: {epochs}")
    lines.append(f"- Training samples: {X_train_proc.shape[0]}")
    lines.append("")
    header = "| setting | " + " | ".join(f"y{t + 1} AUROC" for t in range(len(task_classes))) + " |"
    divider = "|" + " --- |" * (len(task_classes) + 1)
    lines.append(header)
    lines.append(divider)
    for row in table_rows:
        values = " | ".join(f"{row[f'y{t + 1}_auroc']:.4f}" for t in range(len(task_classes)))
        lines.append(f"| {row['setting']} | {values} |")
    lines.append("")

    for result in results:
        setting_info = result["setting"]
        lines.append(f"## {setting_info['label']}")
        lines.append("")
        lines.append(f"- beta: {setting_info['beta']}")
        lines.append(f"- info_config: {setting_info['info_config']}")
        lines.append("")
        for task_name, task_metrics in result["metrics"].items():
            lines.append(f"### {task_name}")
            lines.append("")
            lines.append(
                "- AUROC (macro): "
                f"{task_metrics['auroc_macro']:.4f}; AUROC (micro): {task_metrics['auroc_micro']:.4f}"
            )
            lines.append(
                "- AUPRC (macro): "
                f"{task_metrics['auprc_macro']:.4f}; AUPRC (micro): {task_metrics['auprc_micro']:.4f}"
            )
            lines.append(f"- Accuracy: {task_metrics['acc_top1']:.4f}")
            lines.append(f"- F1 macro: {task_metrics['f1_macro']:.4f}")
            loss_vals = result["losses"][task_name]
            lines.append(
                "- Losses: total={total:.4f}, recon={reconstruction:.4f}, KL={kl:.4f}, "
                "CE={classification:.4f}, info_penalty={info_penalty:.4f}, weight={info_weight:.4f}".format(
                    **loss_vals
                )
            )
            lines.append("")
    md_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SUAVE on the hard benchmark under beta/InfoVAE configurations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/hard_beta_info_results.json"),
        help="Path to the JSON report; a Markdown companion will be generated alongside.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for each model.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=202352,
        help="Seed used when sampling train/test split random_states.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20201021,
        help="Base seed for model initialisation (incremented per setting).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1000,
        help="Optional cap on the number of training samples to speed up InfoVAE runs.",
    )
    args = parser.parse_args()

    settings = [
        ExperimentSetting("beta_1.0", beta=1.0),
        ExperimentSetting("beta_2.0", beta=2.0),
        ExperimentSetting("beta_4.0", beta=4.0),
        ExperimentSetting(
            "info_alpha0.3_lambda1.0",
            beta=1.0,
            info_config=InfoVAEConfig(alpha=0.3, lambda_=1.0),
        ),
        ExperimentSetting(
            "info_alpha0.7_lambda1.3",
            beta=1.0,
            info_config=InfoVAEConfig(alpha=0.7, lambda_=1.3),
        ),
    ]

    run_experiments(
        settings,
        split_seed=args.split_seed,
        epochs=args.epochs,
        base_model_seed=args.model_seed,
        output=args.output,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
