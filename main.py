import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import calculate_metrics
from src.feature_extraction import extract_features
from src.preprocessing import preprocess
from src.utils import get_all_images, load_image
from src.verification import identify_user


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def enroll_database(train_dir: Path) -> dict:
    """Build enrolled user database: person_id -> list of fused feature dicts."""
    database = {}
    for item in get_all_images(str(train_dir)):
        pid = item["id"]
        if pid is None:
            continue
        image = load_image(item["path"])
        if image is None:
            continue
        processed = preprocess(image)
        _, feats = extract_features(processed)
        if feats is None or feats.get("sift") is None:
            continue
        database.setdefault(pid, []).append(feats)
    return database


def identification_gallery(full_database: dict, top_k: int = 2) -> dict:
    """
    Keeps the top_k enrolled templates per identity (by SIFT count) for better
    coverage than a single image, without scanning the full gallery every time.
    """
    gallery = {}
    for pid, templates in full_database.items():
        usable = [t for t in templates if t is not None and t.get("sift") is not None]
        if not usable:
            continue
        ordered = sorted(usable, key=lambda t: len(t["sift"]), reverse=True)
        gallery[pid] = ordered[: max(1, top_k)]
    return gallery


def run_validation(test_dir: Path, database: dict) -> list:
    """
    Open-set identification: each test image is a probe.
    Returns rows for calculate_metrics: {'true_id', 'pred_id'}.
    """
    results = []
    items = [i for i in get_all_images(str(test_dir)) if i["id"] is not None]
    total = len(items)
    for n, item in enumerate(items, start=1):
        true_id = item["id"]
        image = load_image(item["path"])
        if image is None:
            continue
        processed = preprocess(image)
        _, feats = extract_features(processed)
        if feats is None or feats.get("sift") is None:
            results.append({"true_id": true_id, "pred_id": "Unknown"})
            continue
        pred_id, _score = identify_user(feats, database)
        results.append({"true_id": true_id, "pred_id": pred_id})
        if n == 1 or n % 25 == 0 or n == total:
            print(f"  Probes processed: {n}/{total}", flush=True)
    return results


def _rates(metrics: dict) -> dict:
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def plot_evaluation(metrics: dict, out_path: Path) -> None:
    rates = _rates(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    counts = [metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]]
    labels = ["TP", "FP", "TN", "FN"]
    axes[0].bar(labels, counts, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    axes[0].set_title("Confusion counts")
    axes[0].set_ylabel("Count")

    names = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [
        metrics["accuracy"],
        rates["precision"],
        rates["recall"],
        rates["f1"],
    ]
    colors = ["#9467bd", "#17becf", "#bcbd22", "#e377c2"]
    x = np.arange(len(names))
    axes[1].bar(x, vals, color=colors, width=0.65)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy & error rates (identification)")
    for i, v in enumerate(vals):
        axes[1].text(i, min(v + 0.03, 1.0), f"{v:.3f}", ha="center", fontsize=9)

    note = (
        "RootSIFT + histogram fusion; top-2 templates; tuned margin/threshold.\n"
        "Precision = TP/(TP+FP); Recall = TP/(TP+FN)."
    )
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.55,
        note,
        fontsize=10,
        verticalalignment="center",
        family="sans-serif",
    )
    axes[2].text(
        0.02,
        0.2,
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {rates['precision']:.4f}\n"
        f"Recall: {rates['recall']:.4f}\n"
        f"F1: {rates['f1']:.4f}",
        fontsize=11,
        verticalalignment="center",
        family="monospace",
    )

    fig.suptitle("Fingerprint identification — RootSIFT + histogram fusion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    root = _project_root()
    train_dir = root / "data" / "train"
    test_dir = root / "data" / "test"

    if not train_dir.is_dir():
        print(f"Missing training folder: {train_dir}", file=sys.stderr, flush=True)
        return 1
    if not test_dir.is_dir():
        print(f"Missing test folder: {test_dir}", file=sys.stderr, flush=True)
        return 1

    print("Enrolling from train set (RootSIFT + intensity histogram)…", flush=True)
    full_database = enroll_database(train_dir)
    n_templates = sum(len(v) for v in full_database.values())
    print(
        f"  Identities: {len(full_database)}, stored templates: {n_templates}",
        flush=True,
    )

    use_full = os.environ.get("FPS_FULL_GALLERY", "").strip() == "1"
    top_templates = int(os.environ.get("FPS_TOP_TEMPLATES", "2").strip() or "2")
    if use_full:
        identify_db = full_database
        mode = "all enrolled templates per identity"
    else:
        identify_db = identification_gallery(full_database, top_k=top_templates)
        mode = f"top {top_templates} templates per identity (by SIFT count)"
    print(f"Running open-set identification ({mode})…", flush=True)
    print(
        "Note: full train+test pass can take tens of minutes on a laptop CPU.",
        flush=True,
    )
    results = run_validation(test_dir, identify_db)

    metrics = calculate_metrics(results)
    rates = _rates(metrics)
    print(
        "Metrics — "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={rates['precision']:.4f}, "
        f"recall={rates['recall']:.4f}, "
        f"f1={rates['f1']:.4f}, "
        f"TP={metrics['tp']}, FP={metrics['fp']}, "
        f"TN={metrics['tn']}, FN={metrics['fn']}",
        flush=True,
    )

    plot_path = root / "evaluation_plot.png"
    plot_evaluation(metrics, plot_path)
    print(f"Saved chart: {plot_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
