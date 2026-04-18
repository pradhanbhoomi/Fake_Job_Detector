# Fake_Job_Detector
# 🕵️ Fake Job Posting Detector (NLP + BERT)

An AI-powered system to detect fraudulent job postings using Natural Language Processing and Machine Learning.

🔗 **Live App:**  
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.2:8501 
📂 **Dataset Size:** 28,000+ job listings  


## 🚀 Overview

Fake job postings are a growing online fraud problem.  
This project builds an end-to-end machine learning system to classify job listings as:

- 🚨 Fake Job  
- ✅ Legitimate Job  

The system combines traditional NLP techniques with deep learning for accurate detection.


## 🧠 Key Features

- 🔍 Real-time job description analysis  
- 📊 Fraud risk scoring (Low / Medium / High)  
- 🧠 AI-based explanation (important words influencing prediction)  
- ⚖️ Model comparison (TF-IDF vs BERT)  
- 📈 Risk breakdown:
  - Text Risk  
  - Contact Risk  
  - Language Risk  


## 🏗️ Tech Stack

- Python  
- Scikit-learn  
- Transformers (Hugging Face)  
- PyTorch  
- Streamlit  
- Pandas, NumPy  


## 📊 Models Used

### 1️⃣ TF-IDF + Logistic Regression
- Lightweight and fast  
- Uses text features + engineered fraud signals  
- Achieved ~98% accuracy  

### 2️⃣ BERT (bert-base-uncased)
- Deep contextual NLP model  
- Detects subtle scam patterns  
- Improves understanding of job descriptions  


## ⚙️ Feature Engineering

Custom fraud indicators added:
- Email/contact presence  
- Urgency keywords (e.g., "urgent", "earn fast")  
- Suspicious salary mentions  
- WhatsApp/Telegram references  


## 🖥️ Application

Built an interactive **Streamlit web app** where users can:

1. Paste a job description  
2. Select model (TF-IDF or BERT)  
3. Get prediction + fraud probability  
4. View explanation and risk breakdown  


## 📁 Project Structure
fake-job-detector/
├── app.py
├── train.py
├── bert_train.py
├── model.pkl
├── vectorizer.pkl
├── styles.css
├── requirements.txt
├── data/
└── bert_model/


## ▶️ Run Locally

### 1. Clone the repository

git clone (https://github.com/pradhanbhoomi/Fake_Job_Detector)

cd fake-job-detector


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the app

streamlit run app.py



## 🌐 Deployment

Deployed using **Streamlit Cloud** for real-time access.

## 📈 Results

- ~98% accuracy on classification task  
- Strong precision and recall  
- BERT enhances contextual understanding  


## 🔮 Future Improvements

- Real-time scraping from job platforms  
- Explainability using SHAP/LIME  
- API deployment using FastAPI  
- Advanced analytics dashboard  

## 👤 Author

**Bhoomi Pradhan**


## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!
