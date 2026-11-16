"""Streamlit app: draw one Bangla word, get the whole predicted word."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from bangla_common import (LABELS_PATH, MODEL_INFO_PATH, MODEL_PATH, WordRecognizer, load_labels, load_lexicon,
                           nfc, rgba_to_gray)

STUDENT_NAME = "Sumaiya"
STUDENT_ID = "1000129288"

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_recognizer() -> WordRecognizer:
    model = tf.keras.models.load_model(MODEL_PATH)
    return WordRecognizer(model, load_labels(LABELS_PATH), load_lexicon())


def draw_segments(gray: np.ndarray, result: dict) -> Image.Image:
    img = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(img)
    seg = result["segmentation"]
    if seg["headline"]:
        y0, y1 = seg["headline"]
        d.rectangle([0, y0, img.width - 1, y1], outline=(255, 200, 0), width=1)
    if seg["box"]:
        top, bottom = seg["box"][1], seg["box"][3]
        for i, s in enumerate(seg["segments"]):
            colour = (80, 160, 255) if s.bar_prob < 0.5 else (255, 90, 90)
            d.rectangle([s.x0, top, s.x1, bottom], outline=colour, width=2)
            d.text((s.x0 + 2, bottom + 2), str(i + 1), fill=colour)
    return img


st.title("Bangla Handwritten Word Recognition")
st.caption(f"{STUDENT_NAME} · ID {STUDENT_ID} · CNN trained on BanglaLekha-Isolated, tracked with MLflow")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(f"No trained model at {MODEL_PATH}. Run `python train.py --data-dir <dataset folder>` first.")
    st.stop()

recognizer = get_recognizer()

with st.sidebar:
    st.header("Settings")
    stroke = st.slider("Pen width", 6, 20, 10)
    use_lexicon = st.checkbox("Use the Bangla word list", value=True,
                              help="Match the character reading against lexicon.txt to return a full word.")
    extra = st.text_area("Extra words (one per line)", value="",
                         help="Words added here are also used for matching, e.g. your own test words.")
    show_debug = st.checkbox("Show segmentation details", value=True)
    if MODEL_INFO_PATH.exists():
        st.caption(f"Model info: `{MODEL_INFO_PATH.read_text()}`")
    st.caption(f"Words in lexicon: {len(recognizer.lexicon)}")

extra_words = [nfc(w) for w in extra.splitlines() if w.strip()]
recognizer.set_lexicon(load_lexicon() + extra_words)

st.markdown(
    "Write **one Bangla word** in the black box (for example **মানুষ** or **জল**). "
    "Write big, keep the letters side by side, and leave a small gap between letters if you can."
)
canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=stroke, stroke_color="#FFFFFF",
                   background_color="#000000", height=260, width=900, drawing_mode="freedraw",
                   key="word_canvas")
go = st.button("Predict word", type="primary")

if go:
    if canvas.image_data is None:
        st.warning("Draw a word first.")
        st.stop()
    gray = rgba_to_gray(canvas.image_data)
    if gray.max() < 50:
        st.warning("The canvas is empty. Draw a word first.")
        st.stop()
    with st.spinner("Reading..."):
        result = recognizer.recognize(gray, stroke_width=stroke, use_lexicon=use_lexicon)
    if not result["word"]:
        st.warning("No character found. Try drawing bigger.")
        st.stop()

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Predicted word")
        st.markdown(f"<div style='font-size:72px;font-weight:600;line-height:1.2'>{result['word']}</div>",
                    unsafe_allow_html=True)
        st.caption(f"from the {result['source']} · character reading: {result['raw']}")
    with right:
        if result["candidates"]:
            st.subheader("Top matches")
            df = pd.DataFrame(result["candidates"])
            df["score_per_segment"] = df["score_per_segment"].map(lambda v: round(float(np.exp(v)), 3))
            df = df.rename(columns={"score_per_segment": "match score"})
            st.dataframe(df, hide_index=True, use_container_width=True)

    if show_debug:
        st.subheader("How the word was cut")
        st.image(draw_segments(gray, result), caption="yellow = headline (matra), blue = letter, red = vowel bar",
                 use_column_width=True)
        cols = st.columns(max(1, len(result["crops"])))
        for i, (col, crop) in enumerate(zip(cols, result["crops"])):
            col.image(crop, caption=f"segment {i + 1}", width=64)
        st.dataframe(pd.DataFrame([{"segments": f"{p['start'] + 1}-{p['start'] + p['length']}",
                                    "character": p["char"], "confidence": round(p["confidence"], 3)}
                                   for p in result["parts"]]), hide_index=True)
else:
    st.info("Draw a word, then press **Predict word**.")
