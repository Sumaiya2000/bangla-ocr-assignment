#!/usr/bin/env python3
"""Train a Bangla handwritten character CNN on BanglaLekha-Isolated and log the run in MLflow.

The CNN learns the 50 basic Bangla characters (dataset folders 1-50). app.py then
cuts a drawn word into characters and joins them into a whole word.

Examples
    python train.py --data-dir data/BanglaLekha-Isolated --arch small  --epochs 5  --run-name small-cnn
    python train.py --data-dir data/BanglaLekha-Isolated --arch medium --epochs 12 --run-name medium-cnn
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf

from bangla_common import (BASE_DIR, EXPERIMENT_NAME, IMG_SIZE, LABELS_PATH, MLFLOW_ARTIFACTS, MLFLOW_DIR,
                           MODEL_INFO_PATH, MODEL_PATH, MODELS_DIR, TRACKING_URI, default_labels, prepare_glyph,
                           read_dataset_image, save_labels)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BanglaLekha-Isolated character CNN with MLflow tracking.")
    p.add_argument("--data-dir", default=str(BASE_DIR / "data" / "BanglaLekha-Isolated"),
                   help="Folder that contains the class folders 1..84 (searched recursively).")
    p.add_argument("--arch", choices=["small", "medium"], default="medium")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--limit-per-class", type=int, default=0, help="0 = use every image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--always-save", action="store_true",
                   help="Overwrite models/model.keras even if an earlier run was better.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def find_class_dirs(data_dir: Path) -> dict[str, Path]:
    """Find the folders named 1..50 anywhere under data_dir."""
    wanted = {row["folder"] for row in default_labels()}
    found: dict[str, Path] = {}
    for root, dirs, _ in os.walk(data_dir):
        for d in dirs:
            if d in wanted and d not in found:
                found[d] = Path(root) / d
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found), key=int)
    if missing:
        raise SystemExit(
            f"Could not find class folders {missing[:10]}... under {data_dir}.\n"
            "Download BanglaLekha-Isolated from https://data.mendeley.com/datasets/hf6sf8zrkc/2, unzip it, "
            "and pass the folder with --data-dir (it must contain folders named 1, 2, ..., 50)."
        )
    return found


def load_dataset(data_dir: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    cache = BASE_DIR / "data" / "cache" / f"banglalekha_basic_{IMG_SIZE}px_limit{limit}.npz"
    if cache.exists():
        with np.load(cache) as f:
            print(f"Loaded cached arrays from {cache}")
            return f["x"], f["y"]

    class_dirs = find_class_dirs(data_dir)
    xs, ys = [], []
    start = time.time()
    for row in default_labels():
        files = sorted(p for p in class_dirs[row["folder"]].iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if limit:
            files = files[:limit]
        for path in files:
            try:
                xs.append(prepare_glyph(read_dataset_image(path)))
                ys.append(row["index"])
            except Exception as exc:  # a broken file should not stop training
                print(f"skip {path}: {exc}")
        print(f"class {row['folder']:>2} {row['char']}: {len(files)} images  ({time.time() - start:.0f}s)")
    x = np.stack(xs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    print(f"Saved cache {cache}")
    return x, y


def split(x: np.ndarray, y: np.ndarray, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx = np.concatenate(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def stroke_augment(image: tf.Tensor, label: tf.Tensor):
    """Randomly thicken or thin the strokes so canvas pens and real pens both look familiar."""
    img = tf.cast(image, tf.float32)[None, ...]
    r = tf.random.uniform([])
    img = tf.cond(r < 0.35, lambda: tf.nn.max_pool2d(img, 3, 1, "SAME"), lambda: img)
    img = tf.cond(r > 0.85, lambda: -tf.nn.max_pool2d(-img, 2, 1, "SAME"), lambda: img)
    return img[0], label


def make_datasets(x_tr, y_tr, x_va, y_va, batch: int, augment: bool, seed: int):
    x_tr = x_tr[..., None]
    x_va = x_va[..., None]
    train = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(min(len(x_tr), 20000), seed=seed)
    if augment:
        train = train.map(stroke_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train = train.map(lambda a, b: (tf.cast(a, tf.float32), b))
    train = train.batch(batch).prefetch(tf.data.AUTOTUNE)
    val = tf.data.Dataset.from_tensor_slices((x_va.astype("float32"), y_va)).batch(batch).prefetch(tf.data.AUTOTUNE)
    return train, val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(arch: str, num_classes: int, dropout: float, augment: bool) -> tf.keras.Model:
    L = tf.keras.layers
    layers: list = [L.Input(shape=(IMG_SIZE, IMG_SIZE, 1))]
    if augment:  # only active during training
        layers += [L.RandomRotation(0.06, fill_value=0.0), L.RandomZoom(0.1, fill_value=0.0),
                   L.RandomTranslation(0.08, 0.08, fill_value=0.0)]
    layers.append(L.Rescaling(1 / 255.0))  # the saved model takes raw 0-255 pixels

    if arch == "small":
        layers += [
            L.Conv2D(32, 3, activation="relu", padding="same"), L.MaxPooling2D(),
            L.Conv2D(64, 3, activation="relu", padding="same"), L.MaxPooling2D(),
            L.Flatten(), L.Dense(128, activation="relu"), L.Dropout(dropout),
        ]
    else:
        layers += [
            L.Conv2D(32, 3, activation="relu", padding="same"), L.Conv2D(32, 3, activation="relu", padding="same"),
            L.MaxPooling2D(),
            L.Conv2D(64, 3, activation="relu", padding="same"), L.Conv2D(64, 3, activation="relu", padding="same"),
            L.MaxPooling2D(),
            L.Conv2D(128, 3, activation="relu", padding="same"), L.MaxPooling2D(),
            L.Flatten(), L.Dense(256, activation="relu"), L.Dropout(dropout),
        ]
    layers.append(L.Dense(num_classes, activation="softmax"))
    return tf.keras.Sequential(layers, name=f"bangla_{arch}_cnn")


# ---------------------------------------------------------------------------
# Plots for MLflow
# ---------------------------------------------------------------------------
def plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ep = range(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["loss"], "o-", label="train")
    axes[0].plot(ep, history["val_loss"], "o-", label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(ep, history["accuracy"], "o-", label="train")
    axes[1].plot(ep, history["val_accuracy"], "o-", label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("predicted class index")
    ax.set_ylabel("true class index")
    ax.set_title("Validation confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(x: np.ndarray, y: np.ndarray, path: Path) -> None:
    """One training image per class, titled with the folder number (check against labels.json)."""
    fig, axes = plt.subplots(5, 10, figsize=(12, 6.5))
    for c, ax in enumerate(axes.ravel()):
        idx = np.where(y == c)[0]
        if len(idx):
            ax.imshow(x[idx[0]], cmap="gray")
        ax.set_title(f"folder {c + 1}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def get_experiment_id() -> str:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(MLFLOW_ARTIFACTS / EXPERIMENT_NAME).as_uri())


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    device = "GPU" if gpus else "CPU"

    labels = default_labels()
    save_labels(LABELS_PATH)
    x, y = load_dataset(Path(args.data_dir), args.limit_per_class)
    x_tr, y_tr, x_va, y_va = split(x, y, args.val_split, args.seed)
    print(f"train {len(x_tr)} images, validation {len(x_va)} images, {len(labels)} classes, device {device}")

    augment = not args.no_augment
    train_ds, val_ds = make_datasets(x_tr, y_tr, x_va, y_va, args.batch_size, augment, args.seed)
    model = build_model(args.arch, len(labels), args.dropout, augment)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_id = get_experiment_id()
    run_name = args.run_name or f"{args.arch}-{args.epochs}ep-lr{args.lr:g}"

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_params({
            "dataset": "BanglaLekha-Isolated basic characters (folders 1-50)",
            "arch": args.arch, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "dropout": args.dropout, "augment": augment, "image_size": IMG_SIZE, "classes": len(labels),
            "train_images": len(x_tr), "val_images": len(x_va), "limit_per_class": args.limit_per_class,
            "parameters": model.count_params(), "device": device, "seed": args.seed,
        })
        log_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: mlflow.log_metrics(
            {"train_loss": logs["loss"], "train_accuracy": logs["accuracy"],
             "val_loss": logs["val_loss"], "val_accuracy": logs["val_accuracy"]}, step=epoch + 1))
        callbacks = [
            log_epoch,
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ]
        start = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
        seconds = time.time() - start

        probs = model.predict(x_va[..., None].astype("float32"), batch_size=512, verbose=0)
        preds = probs.argmax(axis=1)
        val_acc = float((preds == y_va).mean())
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_va, preds):
            cm[t, p] += 1
        per_class = {labels[c]["char"]: float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(len(labels))}
        mlflow.log_metrics({"final_val_accuracy": val_acc, "train_seconds": seconds})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_history(history.history, tmp / "training_curve.png")
            plot_confusion(cm, tmp / "confusion_matrix.png")
            plot_samples(x_tr, y_tr, tmp / "one_sample_per_class.png")
            (tmp / "per_class_accuracy.json").write_text(json.dumps(per_class, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
            for name in ("training_curve.png", "confusion_matrix.png", "one_sample_per_class.png"):
                mlflow.log_artifact(str(tmp / name), artifact_path="plots")
            mlflow.log_artifact(str(tmp / "per_class_accuracy.json"), artifact_path="metrics")
            run_model = tmp / "model.keras"
            model.save(run_model)
            mlflow.log_artifact(str(run_model), artifact_path="model")
        mlflow.log_artifact(str(LABELS_PATH), artifact_path="model")

        best = json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else {}
        if args.always_save or not MODEL_PATH.exists() or val_acc >= best.get("val_accuracy", 0.0):
            model.save(MODEL_PATH)
            MODEL_INFO_PATH.write_text(json.dumps({
                "run_id": run.info.run_id, "run_name": run_name, "arch": args.arch,
                "val_accuracy": val_acc, "image_size": IMG_SIZE,
            }, indent=2))
            mlflow.set_tag("deployed", "yes")
            print(f"Saved {MODEL_PATH} (val accuracy {val_acc:.4f})")
        else:
            mlflow.set_tag("deployed", "no")
            print(f"Kept the existing model (val accuracy {best['val_accuracy']:.4f} >= {val_acc:.4f})")

        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
        print(f"\nrun {run_name}  id {run.info.run_id}")
        print(f"validation accuracy {val_acc:.4f}  training time {seconds:.0f}s on {device}")
        print("hardest characters:", ", ".join(f"{c} {a:.2f}" for c, a in worst))


if __name__ == "__main__":
    main()
