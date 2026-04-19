import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import calculate_metrics
from src.feature_extraction import extract_features
from src.preprocessing import preprocess
from src.utils import get_all_images, load_image
from src.verification import identify_user


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def enroll_database(train_dir: Path) -> dict:
    """Build enrolled user database: person_id -> list of SIFT descriptor matrices."""
    database = {}
    for item in get_all_images(str(train_dir)):
        pid = item["id"]
        if pid is None:
            continue
        image = load_image(item["path"])
        if image is None:
            continue
        processed = preprocess(image)
        _, des = extract_features(processed)
        if des is None:
            continue
        database.setdefault(pid, []).append(des)
    return database


def identification_gallery(full_database: dict) -> dict:
    """
    One template per identity for 1:N speed.
    Picks the enrolled template with the most SIFT keypoints (richer than list order).
    """
    gallery = {}
    for pid, templates in full_database.items():
        usable = [t for t in templates if t is not None]
        if not usable:
            continue
        best = max(usable, key=len)
        gallery[pid] = [best]
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
        _, des = extract_features(processed)
        if des is None:
            results.append({"true_id": true_id, "pred_id": "Unknown"})
            continue
        pred_id, _score = identify_user(des, database)
        results.append({"true_id": true_id, "pred_id": pred_id})
        if n == 1 or n % 25 == 0 or n == total:
            print(f"  Probes processed: {n}/{total}", flush=True)
    return results


def plot_evaluation(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = [metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]]
    labels = ["TP", "FP", "TN", "FN"]
    axes[0].bar(labels, counts, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    axes[0].set_title("Confusion counts")
    axes[0].set_ylabel("Count")

    axes[1].bar(
        ["Accuracy"],
        [metrics["accuracy"]],
        color="#9467bd",
        width=0.35,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Overall accuracy")
    axes[1].set_ylabel("Accuracy")

    fig.suptitle("Fingerprint identification — evaluation")
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

    print("Enrolling from train set (SIFT)…", flush=True)
    full_database = enroll_database(train_dir)
    n_templates = sum(len(v) for v in full_database.values())
    print(
        f"  Identities: {len(full_database)}, stored templates: {n_templates}",
        flush=True,
    )

    use_full = os.environ.get("FPS_FULL_GALLERY", "").strip() == "1"
    identify_db = full_database if use_full else identification_gallery(full_database)
    mode = "all enrolled templates per identity" if use_full else "one gallery template per identity"
    print(f"Running open-set identification ({mode})…", flush=True)
    print(
        "Note: full train+test pass can take tens of minutes on a laptop CPU.",
        flush=True,
    )
    results = run_validation(test_dir, identify_db)

    metrics = calculate_metrics(results)
    print(
        "Metrics — "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"TP={metrics['tp']}, FP={metrics['fp']}, "
        f"TN={metrics['tn']}, FN={metrics['fn']}",
        flush=True,
    )

    plot_path = root / "evaluation_plot.png"
    plot_evaluation(metrics, plot_path)
    print(f"Saved plot: {plot_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
