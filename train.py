import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import pickle

# LOAD DATASETS (CORRECT)

# IMPORTANT:
# main_dataset.csv = fake_job_postings.csv (has real + fake)
# fake_only_dataset.csv = Fake Postings.csv (only fake)

main_df = pd.read_csv(r"C:\Users\BHOOMI\OneDrive\Desktop\fake-job-detector\data\fake_only_dataset.csv")        # real + fake
fake_df = pd.read_csv(r"C:\Users\BHOOMI\OneDrive\Desktop\fake-job-detector\data\main_dataset.csv")   # only fake

# Fill missing
main_df = main_df.fillna("")
fake_df = fake_df.fillna("")

# LABELS (VERY IMPORTANT)

# main dataset already has both classes
main_df["label"] = main_df["fraudulent"]

# fake dataset → all fake
fake_df["label"] = 1

# DEBUG (keep this for now)
print("\nMain dataset labels:\n", main_df["label"].value_counts())
print("\nFake dataset labels:\n", fake_df["label"].value_counts())

# COMBINE TEXT

def combine_text(df):
    cols = [c for c in ["title", "description", "requirements"] if c in df.columns]
    df["text"] = df[cols].agg(" ".join, axis=1)
    return df

main_df = combine_text(main_df)
fake_df = combine_text(fake_df)

# MERGE DATASETS

df = pd.concat([main_df[["text","label"]], fake_df[["text","label"]]])
df = df.sample(frac=1).reset_index(drop=True)

print("\nFinal dataset size:", df.shape)
print("\nFinal label distribution:\n", df["label"].value_counts())

# CLEAN TEXT

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# FEATURE ENGINEERING (FINAL)

df["has_email"] = df["text"].str.contains("@").astype(int)
df["urgent_words"] = df["text"].str.contains("urgent|immediate|quick money|earn fast").astype(int)
df["has_salary"] = df["text"].str.contains(r"\$|\d+k|\d+ per").astype(int)
df["has_whatsapp"] = df["text"].str.contains("whatsapp|telegram").astype(int)

# MODEL PIPELINE

X_text = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
X_tfidf = vectorizer.fit_transform(X_text)

# IMPORTANT: SAME FEATURES AS app.py
X = hstack([
    X_tfidf,
    df[["has_email","urgent_words","has_salary","has_whatsapp"]]
])

print("\nFeature shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# EVALUATION

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model trained & saved successfully!")