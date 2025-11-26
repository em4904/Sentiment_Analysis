# ------------------------------------------------------------
# Swiggy Review Sentiment Analysis - Flask App (LSTM + GRU)
# ------------------------------------------------------------

from flask import Flask, render_template, request
import numpy as np
import pickle
import re, string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load trained models & preprocessing tools
# -------------------------
lstm_model = load_model("lstm_model.h5")
gru_model = load_model("gru_model.h5")  # ← your GRU model filename

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------
# Helper functions
# -------------------------
max_len = 100

def clean_text(text: str) -> str:
    """Cleans input text similar to training preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(review: str):
    """Predicts sentiment using both LSTM and GRU models."""
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)

    # Get predictions
    lstm_pred = lstm_model.predict(padded)
    gru_pred = gru_model.predict(padded)

    # Get class labels
    lstm_label = label_encoder.inverse_transform([np.argmax(lstm_pred)])[0]
    gru_label = label_encoder.inverse_transform([np.argmax(gru_pred)])[0]

    # Get confidence values
    lstm_conf = float(np.max(lstm_pred) * 100)
    gru_conf = float(np.max(gru_pred) * 100)

    return lstm_label, gru_label, lstm_conf, gru_conf


# -------------------------
# Flask route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    lstm_label = gru_label = ""
    lstm_conf = gru_conf = 0.0

    if request.method == "POST":
        review = request.form["review"]
        lstm_label, gru_label, lstm_conf, gru_conf = predict_sentiment(review)

    return render_template(
        "index.html",
        lstm_label=lstm_label,
        rnn_label=gru_label,       # Keep this variable name for compatibility with your HTML
        lstm_conf=lstm_conf,
        rnn_conf=gru_conf,
    )


# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
