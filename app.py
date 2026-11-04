import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

REPO_ID = "Tawsiful1234/banglat5-medical-checkpoints"
SUBFOLDER = "final-model"

@st.cache_resource
def load_model():
    token = st.secrets["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(REPO_ID, subfolder=SUBFOLDER, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(REPO_ID, subfolder=SUBFOLDER, token=token)
    model.eval()
    return tok, model

st.title("বাংলা মেডিকেল প্রশ্ন সামারাইজার")
tok, model = load_model()

text = st.text_area("রোগীর প্রশ্ন লিখুন", height=200)

if st.button("সামারি তৈরি করো"):
    if text.strip():
        with st.spinner("তৈরি হচ্ছে..."):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=128, num_beams=6,
                                     length_penalty=1.0, early_stopping=True)
            st.success(tok.decode(out[0], skip_special_tokens=True))
    else:
        st.warning("কিছু লিখুন।")
