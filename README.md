
# 🍽️ Sentiment Analysis on Swiggy Reviews  
### 🧠 Deep Learning Project • GRU & LSTM • NLP • Flask Web App  

This project focuses on classifying **customer sentiments from Swiggy food delivery reviews** using deep learning.  
Two recurrent neural network architectures — **GRU** and **LSTM** — were implemented and compared for performance.

The system also includes a **Flask-based web interface** where users can input custom reviews and instantly view sentiment predictions.

---

## 🚀 Features

- 🧠 **Deep Learning Models**
  - GRU-based sentiment classifier  
  - LSTM-based sentiment classifier  

- 🧹 **Text Preprocessing**
  - Lowercasing  
  - Tokenization  
  - Stopword removal  
  - Punctuation cleaning  
  - Padding & sequence generation  

- 📊 **Model Training & Evaluation**
  - Accuracy and loss visualization  
  - Validation split  
  - Confusion matrix  
  - Precision, recall, F1-score  

- 🌐 **Interactive Flask Web App**
  - Simple UI for user review input  
  - Real-time sentiment prediction  
  - Clean, user-friendly interface  

- 📄 **Well-structured Jupyter Notebooks**
  - Exploratory data analysis  
  - Training procedures  
  - Model comparison  


---

## 🧠 Model Architectures

### 🔹 GRU Classifier
- Fast and efficient  
- Performs well with sequential review data  
- Lower training time  

### 🔹 LSTM Classifier
- Better for longer dependencies  
- Slightly higher accuracy  
- Stronger generalization  

---

## 📊 Results & Evaluation

- Both models successfully classified positive, negative, and neutral Swiggy reviews  
- LSTM achieved slightly better accuracy than GRU  
- Confusion matrices and metric reports included in notebooks  
- Real-world review testing validated robustness  

---

## 🌐 Running the Flask Web App

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Start the server

```bash
python app.py
```

### 3️⃣ Open in browser

```
http://127.0.0.1:5000
```

Enter a review and see instant sentiment predictions ⭐

---

## 📦 Installation (Development Setup)

```bash
python -m venv swiggy_env
```

Activate:

```bash
# Windows
swiggy_env\Scripts\activate

# Mac/Linux
source swiggy_env/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* NLTK / spaCy
* Flask
* Matplotlib / Seaborn

---

## 🔮 Future Enhancements

* Add transformer-based model (BERT, DistilBERT)
* Deploy web app using Render / HuggingFace Spaces
* Add multiclass sentiment (anger, joy, disappointment, etc.)
* Add Swiggy-specific sarcasm handling

---

## 📜 License

MIT License.

---

## 🙌 Author

Developed by **Esha** 💛


Just tell me!
```
