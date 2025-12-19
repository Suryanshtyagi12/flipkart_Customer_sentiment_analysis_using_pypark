📊 Customer Review Sentiment Prediction using PySpark

A scalable Sentiment Analysis system built using PySpark MLlib to classify customer reviews into Positive or Negative sentiments.
The project demonstrates how distributed computing can be used to process large-scale textual data efficiently.

🚀 Project Overview

With the rapid growth of e-commerce platforms, understanding customer feedback at scale is critical.
This project uses PySpark, a distributed data processing framework, to analyze Flipkart customer reviews and predict sentiment using a machine learning pipeline.

Key highlights:

Distributed text processing

End-to-end ML pipeline

High accuracy sentiment classification

Scalable and production-ready design

🎯 Objectives

Automatically classify customer reviews into positive or negative

Convert raw text data into numerical features using TF-IDF

Build a scalable ML pipeline using PySpark MLlib

Train and evaluate a Logistic Regression model

Demonstrate cluster computing concepts for real-world datasets

🧠 System Architecture (Workflow)
Customer Reviews (CSV)
        ↓
Data Cleaning & Filtering
        ↓
Text Preprocessing
(Tokenization → Stopword Removal)
        ↓
Feature Engineering
(HashingTF + IDF)
        ↓
Label Encoding
        ↓
Train-Test Split (80:20)
        ↓
Logistic Regression Model
        ↓
Prediction & Evaluation

🛠️ Technology Stack

PySpark (MLlib) – Distributed machine learning

Python 3.11

Apache Spark – Cluster computing engine

Matplotlib – Visualizations

CSV Dataset – Flipkart customer reviews

Tableau – Sentiment dashboard (visual analytics)

📂 Dataset Description

The dataset contains Flipkart product reviews with the following fields:

Column Name	Description
review_text	Customer-written review
sentiment	Review label (positive / negative)
category	Product category

Neutral reviews are removed to keep the task as binary classification.

⚙️ Text Preprocessing Steps

Tokenization
Splits reviews into individual words.

Stop Word Removal
Removes common words like is, the, and.

HashingTF
Converts words into fixed-length numerical vectors.

IDF (Inverse Document Frequency)
Assigns higher weight to meaningful, less frequent words.

🔁 Machine Learning Pipeline

PySpark Pipeline includes:

RegexTokenizer

StopWordsRemover

HashingTF

IDF

StringIndexer

LogisticRegression

Using a pipeline ensures:

Automation

Consistency

Easy scalability and deployment

📈 Model Training

Algorithm: Logistic Regression

Train-Test Split:

80% Training

20% Testing

Logistic Regression is chosen because it:

Performs well for text classification

Handles sparse TF-IDF vectors efficiently

Scales well on distributed systems

✅ Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

🔍 Performance Metrics
Metric	Score
Accuracy	~91%
Precision	~94%
Recall	~91%

These results show strong predictive performance for customer sentiment analysis.

📊 Visualization Dashboard

A Tableau dashboard is created to visualize:

Positive sentiment by category

Negative sentiment by category

Total reviews per category

Average rating trends

This helps businesses gain actionable insights from customer feedback.

📌 Conclusion

This project successfully demonstrates:

Distributed processing using PySpark

Efficient text preprocessing and feature extraction

High-accuracy sentiment classification

Real-world application of Cluster Computing

The system is suitable for large-scale e-commerce sentiment monitoring.

🔮 Future Enhancements

Add Neutral sentiment (3-class classification)

Use Deep Learning models (LSTM, BERT)

Perform hyperparameter tuning

Train on a larger dataset

Deploy using Flask / Streamlit

Enable real-time sentiment prediction

👨‍🎓 Author

Suryansh Tyagi
Course: INT315 – Cluster Computing
School of Computer Science & Engineering
