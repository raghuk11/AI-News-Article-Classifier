"""Train the Embedding + Bidirectional LSTM news classifier on AG News.

Usage:
    python train.py            # full run: 120,000 training articles, up to 5 epochs
    python train.py --quick    # fast demo run on a 20,000-article subset
Outputs go to ./artifacts (model, vocabulary, metrics, figures).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="train on a 20,000-article subset")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    args = ap.parse_args()

    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(C.SEED)

    # 1. load data
    x_all, y_all = C.load_split("train")
    x_test, y_test = C.load_split("test")
    print("Training articles:", len(x_all), "| Test articles:", len(x_test))
    print("Training class counts:", np.bincount(y_all))
    print("Test class counts    :", np.bincount(y_test))

    # 2. shuffle, then hold out 10% of the training data as validation
    idx = np.random.default_rng(C.SEED).permutation(len(x_all))
    if args.quick:
        idx = idx[:20000]
        print("QUICK MODE: using", len(idx), "training articles")
    n_val = int(len(idx) * C.VAL_SPLIT)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    x_tr, y_tr = [x_all[i] for i in tr_idx], y_all[tr_idx]
    x_val, y_val = [x_all[i] for i in val_idx], y_all[val_idx]

    # 3. vocabulary from the TRAINING part only (no leakage from validation/test)
    vec = C.make_vectorizer()
    vec.adapt(tf.data.Dataset.from_tensor_slices(x_tr).batch(256))
    X_tr, X_val, X_test = (C.encode(vec, t) for t in (x_tr, x_val, x_test))
    print("Sequence shape:", X_tr.shape)

    # 4. train
    model = C.build_model()
    model.summary()
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                            restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=[stop])

    # 5. evaluate on the untouched test set
    y_pred = np.argmax(model.predict(X_test, batch_size=256), axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=C.CLASS_NAMES,
                                   digits=4, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print("\nEpochs completed:", len(history.history["loss"]))
    print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 4))
    print("Test-set size:", len(y_test))
    print("Test accuracy:", round(acc, 4))
    print("\n" + classification_report(y_test, y_pred, target_names=C.CLASS_NAMES, digits=4))
    print("Confusion matrix (rows = true, columns = predicted):\n", cm)

    # 6. save artifacts
    model.save(C.MODEL_PATH)
    C.VOCAB_PATH.write_text(json.dumps(vec.get_vocabulary()[2:]), encoding="utf-8")
    metrics = {
        "quick_mode": bool(args.quick),
        "train_articles": int(len(x_tr)),
        "validation_articles": int(len(x_val)),
        "test_articles": int(len(x_test)),
        "epochs_completed": len(history.history["loss"]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, title in ((ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")):
        a.plot(history.history[key], label="train")
        a.plot(history.history["val_" + key], label="validation")
        a.set_title(title); a.set_xlabel("Epoch"); a.legend()
    plt.tight_layout(); plt.savefig(C.ARTIFACT_DIR / "training_curves.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(C.CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(C.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(C.ARTIFACT_DIR / "confusion_matrix.png", dpi=200); plt.close()

    print("\nSaved model, vocabulary, metrics and figures in", C.ARTIFACT_DIR)


if __name__ == "__main__":
    main()
