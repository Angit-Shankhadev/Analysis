"""
app.py
-------
Flask web application that serves the trained sentiment analysis model
for real-time predictions on user-entered text (e.g. a customer review
or social media post).

Run:
    python save_model.py      # trains and saves the model (run once)
    python app.py             # starts the web app on http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib

from preprocessing import preprocess

app = Flask(__name__)

MODEL = joblib.load("../outputs/sentiment_model.joblib")
VECTORIZER = joblib.load("../outputs/tfidf_vectorizer.joblib")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("review_text", "")
    cleaned = preprocess(review_text)
    vec = VECTORIZER.transform([cleaned])

    prediction = MODEL.predict(vec)[0]
    proba = MODEL.predict_proba(vec)[0]
    class_index = list(MODEL.classes_).index(prediction)
    confidence = round(proba[class_index] * 100, 2)

    return render_template(
        "result.html",
        original_text=review_text,
        sentiment=prediction,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)
