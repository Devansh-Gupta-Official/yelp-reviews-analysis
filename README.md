# Yelp Reviews Analysis

## Overview
This Jupyter Notebook analyzes Yelp reviews using Python libraries like Pandas, Matplotlib, Seaborn, NLTK, and Scikit-learn. The analysis includes data visualization, text preprocessing (removing punctuation, stopwords), and implementing a Naive Bayes classifier.

## Contents
**1. Importing Libraries and Dataset**
- Libraries: The notebook starts by importing essential libraries such as Pandas, NumPy, Matplotlib, Seaborn, and NLTK.
- Dataset: It loads the Yelp dataset ('yelp.csv') into a Pandas DataFrame (df) to begin the analysis.
**2. Visualizing the Dataset**
- Data Overview: Describes the dataset structure, columns, and data types (df.info() and df.describe()).
- Text Length Distribution: Visualizes the distribution of review text lengths through histograms and statistical summary (df['length'].describe()).
- Review Analysis by Star Rating: Examines the distribution of reviews across different star ratings using count plots and histograms.
**3. Creating Testing and Training Dataset**
- Text Preprocessing: Demonstrates techniques for text preprocessing, including punctuation removal and stopwords elimination using NLTK.
- Count Vectorizer: Illustrates the process of converting text data into numerical format (bag-of-words) using Scikit-learn's CountVectorizer.
**4. Training the Model**
- Naive Bayes Classifier: Implements a Multinomial Naive Bayes classifier using Scikit-learn to train the model on the preprocessed Yelp reviews.
**5. Evaluating the Model**
- Model Performance: Evaluates the trained model's performance metrics, including confusion matrices and classification reports for both training and test datasets.
**6. Additional Feature: TF-IDF**
- TF-IDF Transformation: Introduces Term Frequency-Inverse Document Frequency as an enhancement to feature representation for text data.
- Model Refitting: Retrains the model using TF-IDF transformed data and evaluates its performance.
