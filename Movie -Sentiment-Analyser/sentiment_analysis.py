import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gradio as gr
import io
from PIL import Image

# Load and preprocess dataset
df = pd.read_csv("idmb-dataset.csv")
df = df[['Overview', 'Rating']].dropna()
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])
df['label'] = df['Rating'].apply(lambda x: 1 if x >= 7.0 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Overview'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Prediction function
def predict_sentiment(review):
    if not review.strip():
        return "<div style='color:gray; font-size: 18px;'>Please enter a valid review.</div>", None

    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😠"
    color = "#2ecc71" if pred == 1 else "#e74c3c"

    result_html = f"""
    <div style='text-align:center; padding:15px; background-color:{color}; color:white;
         font-size:20px; font-weight:bold; border-radius:10px; max-width:250px; margin:auto;'>
        {sentiment}<br><span style='font-size:16px;'>(Confidence: {prob[pred]*100:.2f}%)</span>
    </div>
    """

    # Pie chart
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    labels = ['Negative 😠', 'Positive 😊']
    colors = ['#e74c3c', '#2ecc71']
    ax.pie(prob, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)

    return result_html, image

# Gradio Interface
with gr.Blocks(theme="soft") as interface:
    gr.Markdown("""
    # 🎬 Movie Overview Sentiment Analyzer  
    Predict movie sentiment from its description using Logistic Regression + TF-IDF.
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            review_input = gr.Textbox(label="📝 Movie Overview", placeholder="Paste movie overview...", lines=4)
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")
        with gr.Column(scale=1, min_width=300):
            sentiment_output = gr.HTML(label="🔍 Sentiment Result")
            chart_output = gr.Image(label="📊 Confidence Pie", type="pil", width=200, height=200)

    submit_btn.click(fn=predict_sentiment, inputs=review_input, outputs=[sentiment_output, chart_output])
    clear_btn.click(lambda: ("", None), outputs=[sentiment_output, chart_output])

interface.launch()
