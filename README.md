# 📧 Email Spam Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-NLTK-green)

## 📝 Project Overview
This repository contains a Machine Learning project designed to classify emails as **Spam** or **Ham (Not Spam)**. By leveraging Natural Language Processing (NLP) techniques and classification algorithms, this project efficiently detects unwanted, promotional, or malicious emails and separates them from legitimate ones.

## 🚀 Features
- **Text Preprocessing:** Cleaning email text by removing punctuation, converting to lowercase, tokenization, stop-word removal, and stemming/lemmatization.
- **Feature Extraction:** Converting raw text data into numerical features using techniques like `TF-IDF Vectorizer` or `CountVectorizer`.
- **Machine Learning Models:** Training and evaluating algorithms such as **Multinomial Naive Bayes**, **Logistic Regression**, or **Support Vector Machines (SVM)**.
- **Performance Evaluation:** Measuring success through Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn, NLTK (Natural Language Toolkit)
- **Data Visualization:** Matplotlib, Seaborn

## 📂 Dataset
The model is trained on a labeled dataset containing text from various emails and SMS messages. 
> **Note:** Place your downloaded dataset (e.g., `spam.csv`) into the root directory or a `data/` folder before running the scripts.

## ⚙️ Installation & Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul-1419/Email_Spam_Detection.git
   cd Email_Spam_Detection

2. **Create a virtual environment (Recommended):**
   ```bash
   conda create venv python=3.10 -y
   
3. **Install the required dependencies:**
   ```bash
   pip install requirements.txt

4. **Run the Project:**

📊 Evaluation & Results

Algorithm Used: Multinomial Naive Bayes
Accuracy: ~96%
Precision: Highly prioritized to minimize false positives (we don't want important emails going to the spam folder!).

