# 📧 Spam Classifier

A simple machine learning project that classifies text messages as **Spam** or **Not Spam (Ham)** using Python and Scikit-learn.

This project is part of my learning journey into machine learning and NLP. It uses a dataset of SMS messages to train a basic Naive Bayes classifier.

---

## 🚀 Features

- Classifies SMS messages into "spam" or "ham"
- Text preprocessing using `CountVectorizer`
- Model training using `MultinomialNB` (Naive Bayes)
- Simple accuracy evaluation

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter / PyCharm (for development)

---

## 📂 Dataset

The dataset used is a CSV file containing two main columns:
- `label` - indicates if a message is spam or ham
- `message` - the SMS text content

The dataset was cleaned to remove unnecessary columns like `v1`, `v2`, etc.

---

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/dumilanikonyani/spam_classifier.git
   cd spam_classifier
2. Install dependencies (ideally in a virtual environment):
    pip install -r requirements.txt
3. Run the Script
     python spam_classifier.py
  
