
# 🎬 Movie Sentiment Analyzer using Machine Learning

A web-based sentiment analysis tool that classifies movie overviews or reviews as **Positive 😊** or **Negative 😠** using machine learning techniques.

---

## 🚀 Features

- Enter any movie overview or summary text to predict sentiment.
- Displays confidence score and a pie chart visualization.
- Logistic Regression model trained using TF-IDF vectorization.
- Clean, interactive UI powered by **Gradio**.
- Handles edge cases and empty inputs gracefully.

---

## 🧠 How It Works

1. **Dataset**:
   - Used IMDb dataset containing `Overview` and `Rating` columns.
   - Sentiment label is derived:
     - Rating ≥ 7 → Positive
     - Rating < 7 → Negative

2. **Text Vectorization**:
   - TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors.

3. **Model**:
   - Logistic Regression is trained on the vectorized overviews.

4. **Prediction**:
   - New inputs are vectorized and passed to the model to predict sentiment.
   - Confidence scores are displayed via a pie chart.

---

## 🛠 Project Structure

```
MovieSentimentAnalyzer/
│
├── app.py                 # Main Gradio app and ML logic
├── IDMB Dataset.csv       # IMDb dataset with movie overviews and ratings
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 💻 Run the App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
python app.py
```

The Gradio interface will open in your browser automatically.

---

## ✅ Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- gradio
- pillow

---

## 🗂 Dataset Structure

Make sure your CSV file (`IDMB Dataset.csv`) contains:

- `Overview`: Text summary of the movie.
- `Rating`: Numeric rating (used to label sentiment).

---

## 📊 Output

- Predicted Sentiment: Positive 😊 / Negative 😠
- Confidence Score: Displayed as percentage.
- Pie Chart: Visual confidence representation.

---

## 👨‍💻 Author

Developed by PALLETI YONA  
This project is ideal for learning basic NLP and model deployment techniques.

