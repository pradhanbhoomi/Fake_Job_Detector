import streamlit as st
import pickle
import re
import numpy as np
from scipy.sparse import hstack

# BERT
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# LOAD CSS

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)

st.set_page_config(page_title="AI Job Fraud Detector", page_icon="🕵️")

load_css()

# LOAD MODELS

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

bert_model = BertForSequenceClassification.from_pretrained("bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("bert_model")
bert_model.eval()

# CLEAN TEXT

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# HERO SECTION

st.markdown("""
<h1>🕵️ AI Job Fraud Detector</h1>
<p style='color:#94a3b8; font-size:18px;'>
Detect fake job postings using Machine Learning + BERT
</p>
""", unsafe_allow_html=True)

# MODEL SELECTOR

model_choice = st.radio("Choose Model", ["TF-IDF (Fast)", "BERT (Advanced)"])

# INPUT SECTION (GLASS CARD)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("📄 Analyze Job Posting")
text = st.text_area("", height=200, placeholder="Paste job description here...")

st.markdown('</div>', unsafe_allow_html=True)

# PREDICTION

if st.button("Analyze Job"):

    if text.strip() == "":
        st.warning("Enter job description")
    else:

        # TF-IDF MODEL
        if model_choice == "TF-IDF (Fast)":

            cleaned = clean_text(text)

            has_email = int("@" in text)
            urgent_words = int(any(word in text.lower() for word in ["urgent","immediate","quick money","earn fast"]))
            has_salary = int(bool(re.search(r"\$|\d+k|\d+ per", text.lower())))
            has_whatsapp = int(any(word in text.lower() for word in ["whatsapp","telegram"]))

            X_tfidf = vectorizer.transform([cleaned])

            X = hstack([
                X_tfidf,
                [[has_email, urgent_words, has_salary, has_whatsapp]]
            ])

            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]

        # BERT MODEL
        else:

            inputs = bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            prob = probs[0][1].item()
            prediction = int(prob > 0.5)

        # RESULT (GLASS CARD)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="error-box">
            🚨 <b>Fake Job Detected</b><br>
            Risk Score: {round(prob*100,2)}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            ✅ <b>Legitimate Job</b><br>
            Confidence: {round((1-prob)*100,2)}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # RISK BREAKDOWN

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.subheader("📊 Risk Breakdown")

        text_risk = prob * 100
        contact_risk = 50 if "@" in text else 0
        language_risk = 100 if "urgent" in text.lower() else 20

        col1, col2, col3 = st.columns(3)

        col1.metric("Text Risk", f"{round(text_risk)}%")
        col2.metric("Contact Risk", f"{contact_risk}%")
        col3.metric("Language Risk", f"{language_risk}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # MODEL COMPARISON

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.subheader("⚖️ Model Comparison")

        # TF-IDF again
        cleaned = clean_text(text)
        X_tfidf = vectorizer.transform([cleaned])
        X = hstack([X_tfidf, [[0,0,0,0]]])
        tfidf_prob = model.predict_proba(X)[0][1]

        # BERT again
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        bert_prob = probs[0][1].item()

        col1, col2 = st.columns(2)

        col1.metric("TF-IDF Risk", f"{round(tfidf_prob*100,2)}%")
        col2.metric("BERT Risk", f"{round(bert_prob*100,2)}%")

        st.markdown('</div>', unsafe_allow_html=True)