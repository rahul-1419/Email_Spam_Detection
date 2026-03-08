# 📧 Email Spam Detection using Machine Learning & NLP

## 📌 Project Overview
This project builds an Email Spam Detection system using Natural Language Processing (NLP) and Machine Learning algorithms. Text messages are processed, cleaned, and transformed into numerical features to classify emails as Spam or Ham (Not Spam).

---

## 📊 Dataset Description
The dataset contains labeled email/SMS messages with:

- **Target**: Spam or Ham label  
- **Text**: Message content  

---

## 🔍 Exploratory Data Analysis (EDA)
Key analyses performed:

- Spam vs Ham distribution visualization  
- Text length and sentence pattern analysis  
- Correlation analysis between text features  
- Word frequency analysis for spam and ham messages  
- WordCloud visualization of common spam and ham terms  

---

## 🧹 Data Preprocessing & NLP
- Removed duplicate and irrelevant columns  
- Label encoding of target variable  
- Text cleaning (lowercasing, stopword removal, punctuation removal)  
- Lemmatization and stemming  
- Feature engineering:
  - Character count
  - Word count
  - Sentence count

---

## 🔤 Text Vectorization
- Applied **TF-IDF Vectorization** to convert text into numerical features  
- Limited to top 3000 important words  

---

## 🤖 Machine Learning Models Tested
Multiple classification algorithms were trained and compared:

- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Bagging Classifier
- Extra Trees
- Gradient Boosting

---

## 📈 Model Evaluation
Models evaluated using:

- Accuracy Score  
- Precision Score  
- Confusion Matrix  

Multinomial Naive Bayes achieved strong performance for spam detection.

---

## 💾 Model Files
- Saved TF-IDF Vectorizer (`vectorizer.pkl`)
- Saved Trained Model (`model.pkl`)

---

## 🛠 Tech Stack
**Language:** Python  
**Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
**NLP Tools:** NLTK  
**Machine Learning:** Scikit-learn  
**Model Saving:** Pickle  

---

## 🚀 Project Workflow
Data Cleaning → NLP Processing → Feature Engineering → TF-IDF → Model Training → Evaluation → Model Saving

---

## 🎯 Business Use Case
- Automated spam email filtering  
- Improves inbox management systems  
- Reduces phishing and unwanted message risks  

---

## 🏷 Tags
`Machine Learning` `NLP` `Spam Detection` `Text Classification` `Python`
