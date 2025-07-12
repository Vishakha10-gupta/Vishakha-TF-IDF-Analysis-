import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data
texts = [
    "Diabetes affects blood sugar levels",
    "Stock market is volatile today",
    "Insulin is prescribed for diabetes",
    "Inflation impacts investor confidence",
    "Blood pressure is rising",
    "Economy is slowing down"
]
labels = ["health", "finance", "health", "finance", "health", "finance"]

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# Accuracy by keyword
def keyword_accuracy():
    keywords = ["diabetes", "stock", "insulin", "inflation", "blood", "economy"]
    results = []
    for kw in keywords:
        matches = [i for i, t in enumerate(texts) if kw in t.lower()]
        if matches:
            X_kw = vectorizer.transform([texts[i] for i in matches])
            y_kw = [labels[i] for i in matches]
            y_pred = model.predict(X_kw)
            acc = accuracy_score(y_kw, y_pred)
            results.append(f"{kw}: {acc*100:.2f}%")
        else:
            results.append(f"{kw}: No samples")
    return "\n".join(results)

# Gradio interface
def classify(text):
    pred = model.predict(vectorizer.transform([text]))[0]
    accs = keyword_accuracy()
    return f"Prediction: {pred}\n\nKeyword Accuracies:\n{accs}"

gr.Interface(fn=classify, inputs="text", outputs="text", title="TF-IDF Classifier").launch()
