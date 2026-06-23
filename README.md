# Airbnb Sentiment Analysis

An NLP pipeline for sentiment classification of Airbnb reviews, built as part of a data science portfolio project.

## Project Overview

This project classifies guest reviews as **positive**, **neutral**, or **negative** 
using a multistage NLP pipeline. The final model (Linear SVC) achieves 91% accuracy 
on a balanced dataset of 258,000+ English reviews.

## Pipeline

| Notebook | Description |
|----------|-------------|
| 01_language_detection | Filters 342,904 reviews to English only |
| 02_text_cleaning | Negation handling, lemmatization, feature extraction |
| 03_sentiment_labeling | VADER-based sentiment labeling |
| 04_eda_visualizations | Distribution, wordclouds, timeline analysis |
| 05_ml_models | TF-IDF + 4 ML models compared |
| 06_deep_learning | Bidirectional LSTM with SMOTE |
| 07_llm | DistilBERT evaluation and model comparison |

## Results

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|-------|----------|-------------|------------|-------------|
| **Linear SVC** ✅ | **91%** | **0.94** | **0.87** | **0.93** |
| Logistic Regression | 90% | 0.91 | 0.86 | 0.93 |
| XGBoost | 90% | 0.92 | 0.86 | 0.92 |
| Random Forest | 89% | 0.94 | 0.85 | 0.89 |
| LSTM | 86% | 0.85 | 0.83 | 0.90 |
| DistilBERT | N/A | N/A | N/A | N/A |

## Tech Stack

`Python` `scikit-learn` `TensorFlow` `HuggingFace` `NLTK` `VADER` `Streamlit`

## Dashboard

The interactive dashboard allows users to explore the dataset and 
predict the sentiment of any review in real time.

## Live Demo

![Streamlit](https://ltrickie23-airbnb-sentiment-analysis-appstreamlit-app-jty6ma.streamlit.app/)

## Author
Jorge Nava - Mechatronics Engineer | Data Science & Machine Learning.
[LinkedIn](https://www.linkedin.com/in/jorgenavao)
