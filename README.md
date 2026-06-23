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

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ltrickie23-airbnb-sentiment-analysis-appstreamlit-app-jty6ma.streamlit.app/)

## Author
Jorge Nava - Mechatronics Engineer | Data Science & Machine Learning.
[![linkedin]([app-generated-url](https://linkedin-github-readme.onrender.com/api/render/Jorge%20Nava/Mechatronics%20Engineer%20|%20Data%20Science%20&%20Machine%20Learning/Maintenance%20Engineer/Bachelors%20Degree/light/https%3A%2F%2Fmedia.licdn.com%2Fdms%2Fimage%2Fv2%2FD4E03AQEdjk3fk_4nvw%2Fprofile-displayphoto-scale_400_400%2FB4EZ5mOvxeHcAg-%2F0%2F1779831604801%3Fe%3D1783555200%26v%3Dbeta%26t%3DM6L3ZNNc6SN0A_Qfw5mr5dMriXiY_5Pwr8ypGS3_0Pw))]((https://www.linkedin.com/in/jorgenavao))
