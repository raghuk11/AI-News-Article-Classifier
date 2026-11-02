# News Article Classification using Deep Learning

Mini project — SRM TRP Engineering College, Department of Artificial Intelligence & Data Science (2026-2027).

Team: MOHAN KARTHIK S (814724243094), RAGHU K (814724243112), RITHESH P (814724243117)
Guide: Mrs. K. Cinthamani

A deep-learning prototype that classifies English news articles into four topics —
**World, Sports, Business, Sci/Tech** — using word embeddings and a Bidirectional LSTM
trained on the public **AG News** dataset (TensorFlow / Keras). A small Streamlit web app
lets you paste a headline and see the predicted topic and class probabilities.

## Project structure

```
news_article_classification/
├── common.py            shared settings, data loading, preprocessing, model definition
├── train.py             trains the model and saves model, vocabulary, metrics and figures
├── predict.py           command-line prediction
├── app.py               Streamlit web prototype
├── sample_headlines.txt example headlines
├── requirements.txt     Python packages
├── artifacts/           created by train.py (model, vocab.json, metrics.json, PNG figures)
└── docs/                the 12-page mini project report (.docx and .pdf)
```

## 1. Set up

Use Python 3.9 – 3.12 (TensorFlow does not yet support every newest Python version).

```bash
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py --quick     # fast demo (20,000-article subset) - good for a first test
python train.py             # full run: 120,000 training articles, up to 5 epochs
```

The AG News dataset downloads automatically the first time (internet needed).
Training prints the dataset counts, model summary, validation accuracy, **test accuracy,
per-class precision/recall/F1 and the confusion matrix**, and saves everything to `artifacts/`:
`news_bilstm.keras`, `vocab.json`, `metrics.json`, `training_curves.png`, `confusion_matrix.png`.

No GPU? Use Google Colab: upload the folder, then run
`!pip install -r requirements.txt` and `!python train.py`.

## 3. Run the prototype

```bash
streamlit run app.py        # opens http://localhost:8501
python predict.py "Stocks rally as central bank holds interest rates steady"
python predict.py --file sample_headlines.txt
```

## 4. Complete the report

Open `docs/News_Article_Classification_Mini_Project_Report.docx` and replace every
"[Fill after test]" and "[ ]" field (Sections 4.2 and 4.3) with the values printed by
`train.py` / stored in `artifacts/metrics.json`. Add `training_curves.png`,
`confusion_matrix.png` and screenshots of the app. Report only measured values.

## Model summary

| Item | Value |
|---|---|
| Dataset | AG News (`ag_news_subset`): 120,000 train / 7,600 test, 4 balanced classes |
| Preprocessing | TextVectorization, 20,000-word vocabulary (training text only), 100 tokens |
| Network | Embedding(128) → BiLSTM(64) → Dropout(0.5) → Dense(64, ReLU) → Dense(4, softmax) |
| Training | Adam (lr 0.001), batch 128, up to 5 epochs, early stopping (patience 2), 10% validation, seed 42 |

## Notes and limitations

- This is an educational baseline. It handles English text and four fixed topics only,
  cannot verify facts or detect misinformation, and should not be used for editorial decisions
  without human review. Business and Sci/Tech articles are the most likely to be confused.
- Check the AG News terms of use before redistributing the data; this repository does not include it.
- Troubleshooting: if TensorFlow will not install, check your Python version (3.9–3.12);
  if `tfds` cannot download, check your internet/proxy; if you run out of memory, use
  `--batch-size 64` or `--quick`.
