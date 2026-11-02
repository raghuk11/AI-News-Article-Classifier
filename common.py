"""Shared settings and helpers for the News Article Classification project."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# ---------------- configuration ----------------
SEED = 42
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
VOCAB_SIZE = 20000
SEQ_LEN = 100
EMBED_DIM = 128
LSTM_UNITS = 64
DROPOUT = 0.5
BATCH_SIZE = 128
EPOCHS = 5
VAL_SPLIT = 0.10

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "news_bilstm.keras"
VOCAB_PATH = ARTIFACT_DIR / "vocab.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


# ---------------- data ----------------
def load_split(split):
    """Load an AG News split; headline + description become one text string."""
    rows = list(tfds.as_numpy(tfds.load("ag_news_subset", split=split)))
    texts = [(r["title"] + b". " + r["description"]).decode("utf-8") for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    return texts, labels


# ---------------- preprocessing ----------------
def make_vectorizer(vocabulary=None):
    """TextVectorization layer. Pass a saved vocabulary (without the two special
    tokens) for inference; otherwise call .adapt() on TRAINING text only."""
    return layers.TextVectorization(max_tokens=VOCAB_SIZE,
                                    output_sequence_length=SEQ_LEN,
                                    vocabulary=vocabulary)


def encode(vectorizer, texts):
    return vectorizer(tf.constant(list(texts))).numpy()


# ---------------- model ----------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN,), dtype="int64"),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dropout(DROPOUT),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------- inference ----------------
def artifacts_exist():
    return MODEL_PATH.exists() and VOCAB_PATH.exists()


def load_artifacts():
    """Return (model, vectorizer) saved by train.py."""
    model = tf.keras.models.load_model(MODEL_PATH)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return model, make_vectorizer(vocab)


def predict_texts(model, vectorizer, texts):
    """Return an array of shape (n, 4) with class probabilities."""
    return model.predict(encode(vectorizer, texts), verbose=0)
