import os
import sys

import joblib
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    DATASET_DIR,
    LABEL_ENCODER_FILE,
    METADATA_DIR,
    PYTHON_MODEL_DIR,
    SUPPORTED_EXTENSIONS,
)
from feature_extraction.logmel import FRAMES, MELS, SR, logmel, windows

PER_CLIP = 5
SEED = 42

TRAIN_DIRS = [DATASET_DIR / "training", DATASET_DIR / "augmented_training"]
VAL_DIRS = [DATASET_DIR / "validation"]
TEST_DIRS = [DATASET_DIR / "testing"]


def labels():
    enc = joblib.load(PYTHON_MODEL_DIR / LABEL_ENCODER_FILE)
    return [str(c) for c in enc.classes_]


def files(dirs, names):
    out = []
    for d in dirs:
        for i, name in enumerate(names):
            p = os.path.join(d, name)
            if not os.path.isdir(p):
                continue
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    out.append((os.path.join(p, f), i))
    return out


def build(dirs, names, per_clip=PER_CLIP):
    rows = files(dirs, names)
    x, y = [], []
    bad = 0

    for n, (path, index) in enumerate(rows, 1):
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            bad += 1
            continue

        for w in windows(audio, per_clip):
            x.append(logmel(w))
            y.append(index)

        if n % 500 == 0 or n == len(rows):
            print(n, "/", len(rows), "files,", len(x), "windows", flush=True)

    x = np.asarray(x, dtype=np.float32)[..., None]
    y = np.asarray(y, dtype=np.int32)

    if bad:
        print("skipped", bad, "unreadable files")

    return x, y


def cache_path(name):
    return METADATA_DIR / f"browser_cnn_{name}.npz"


def save(name, x, y):
    os.makedirs(METADATA_DIR, exist_ok=True)
    np.savez_compressed(cache_path(name), x=x, y=y)
    print("cached", len(x), "windows to", cache_path(name))


def load(name):
    with np.load(cache_path(name)) as d:
        return d["x"], d["y"]


def get(name, dirs, names, per_clip=PER_CLIP):
    if cache_path(name).exists():
        x, y = load(name)
        print("loaded", len(x), "windows from cache for", name)
        return x, y
    x, y = build(dirs, names, per_clip)
    save(name, x, y)
    return x, y


def make_model(n):
    m = tf.keras.Sequential([tf.keras.layers.Input((FRAMES, MELS, 1))])

    for f in (32, 64, 128, 128):
        m.add(tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False))
        m.add(tf.keras.layers.BatchNormalization())
        m.add(tf.keras.layers.ReLU())
        m.add(tf.keras.layers.MaxPooling2D(2))

    m.add(tf.keras.layers.GlobalAveragePooling2D())
    m.add(tf.keras.layers.Dropout(0.3))
    m.add(tf.keras.layers.Dense(n, activation="softmax"))
    return m


class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.x, verbose=0).argmax(axis=1)
        score = f1_score(self.y, p, average="macro", zero_division=0)
        logs["val_macro_f1"] = score
        print(f"  val_macro_f1 {score:.4f}", flush=True)


def class_weights(y, n):
    counts = np.bincount(y, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n * counts)
    return {i: float(w[i]) for i in range(n)}


def train(xt, yt, xv, yv, n, epochs=60, batch=64, patience=8):
    tf.keras.utils.set_random_seed(SEED)

    model = make_model(n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=patience,
        restore_best_weights=True, verbose=1,
    )

    history = model.fit(
        xt, yt,
        validation_data=(xv, yv),
        epochs=epochs,
        batch_size=batch,
        class_weight=class_weights(yt, n),
        callbacks=[MacroF1(xv, yv), stop],
        verbose=2,
    )

    return model, history
