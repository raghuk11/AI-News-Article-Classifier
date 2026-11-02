"""Streamlit prototype: paste a news headline/article and see the predicted topic.

    streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

import common as C

st.set_page_config(page_title="News Article Classifier", page_icon="📰", layout="centered")
st.title("📰 News Article Classification")
st.caption("Embedding + Bidirectional LSTM trained on AG News (World, Sports, Business, Sci/Tech). "
           "Educational prototype — not an editorial decision tool.")

if not C.artifacts_exist():
    st.error("No trained model found. Run `python train.py --quick` (or `python train.py`) "
             "in this folder first, then reload this page.")
    st.stop()


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return C.load_artifacts()


model, vectorizer = get_model()

EXAMPLES = {
    "— choose an example —": "",
    "Business": "Stocks rally as central bank holds interest rates steady",
    "Sports": "Striker scores twice as the home side wins the championship final",
    "Sci/Tech": "Researchers unveil a new chip that speeds up machine learning",
    "World": "Leaders meet to discuss ceasefire talks and humanitarian aid",
}
choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("News headline and/or short description", value=EXAMPLES[choice],
                    height=140, placeholder="Paste an English news headline or short article…")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        probs = C.predict_texts(model, vectorizer, [text])[0]
        k = int(np.argmax(probs))
        st.subheader(f"Predicted topic: {C.CLASS_NAMES[k]}")
        st.metric("Confidence", f"{probs[k] * 100:.1f}%")
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=C.CLASS_NAMES))
        if probs[k] < 0.60:
            st.info("Low confidence — the article may fit more than one topic. "
                    "A person should review this prediction.")

if C.METRICS_PATH.exists():
    with st.expander("Model evaluation (from the last training run)"):
        m = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
        if m.get("quick_mode"):
            st.warning("This model was trained in quick mode on a small subset.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{m['test_accuracy'] * 100:.2f}%")
        c2.metric("Macro F1", f"{m['macro_f1']:.3f}")
        c3.metric("Test articles", f"{m['test_articles']:,}")
        st.dataframe(pd.DataFrame(m["classification_report"]).T.loc[C.CLASS_NAMES +
                     ["macro avg"], ["precision", "recall", "f1-score"]].round(3))
        st.write("Confusion matrix (rows = true, columns = predicted)")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=C.CLASS_NAMES, columns=C.CLASS_NAMES))
