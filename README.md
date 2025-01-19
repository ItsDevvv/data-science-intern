# Data Science Internship Tasks

This repository contains the solutions for the Data Science Internship tasks assigned with a submission deadline of **16th January 2024**. Each task involves practical applications of data analysis, machine learning, and implementation of models from scratch using Python. The tasks demonstrate proficiency in data science concepts, including exploratory data analysis, sentiment analysis, fraud detection, and regression modeling.

---

## Tasks Overview

### **Task 1: EDA and Visualization of a Real-World Dataset**
- **Objective**: Perform exploratory data analysis (EDA) on a dataset like the Titanic Dataset or Airbnb Listings Dataset.
- **Steps**:
  - Load the dataset and explore its structure.
  - Handle missing values, remove duplicates, and manage outliers.
  - Visualize categorical and numerical variables using bar charts, histograms, and heatmaps.
  - Summarize findings and insights.
- **Outcome**: A detailed Jupyter Notebook or Python script containing EDA, visualizations, and insights.

---

### **Task 2: Text Sentiment Analysis**
- **Objective**: Build a sentiment analysis model using a dataset such as IMDB Reviews.
- **Steps**:
  - Preprocess text: Tokenization, stopword removal, and lemmatization.
  - Convert text into numerical features using TF-IDF or word embeddings.
  - Train a classifier like Logistic Regression or Naive Bayes.
  - Evaluate the model using precision, recall, and F1-score.
- **Outcome**: A working Python script that processes text input, predicts sentiment, and evaluates model performance.

---

### **Task 3: Fraud Detection System**
- **Objective**: Develop a fraud detection system using a dataset like the Credit Card Fraud Dataset.
- **Steps**:
  - Handle imbalanced data using SMOTE or undersampling.
  - Train a Random Forest or Gradient Boosting model for fraud detection.
  - Evaluate precision, recall, and F1-score.
  - Create a command-line interface for testing.
- **Outcome**: A Python script capable of detecting fraudulent transactions with evaluation metrics and a simple testing interface.

---

### **Task 4: Predicting House Prices Using the Boston Housing Dataset**
- **Objective**: Build regression models to predict house prices.
- **Steps**:
  - Normalize numerical features and preprocess categorical variables.
  - Implement Linear Regression, Random Forest, and XGBoost models from scratch.
  - Compare models using RMSE and \( R^2 \) metrics.
  - Visualize feature importance for tree-based models.
- **Outcome**: A Python script implementing regression models, with performance comparisons and visualizations.

---

## Submission Requirements

### 1. **GitHub Repository**
   - All code, datasets, and related files should be organized and pushed to a GitHub repository.
   - Include a link to the repository.

### 2. **Visuals Submission**
   - Record a short video or take screenshots of:
     - Data visualizations
     - Model insights and performance metrics

### 3. **Documentation**
   - Include this `README.md` in your repository, detailing:
     - Steps for each task
     - How to run the scripts
     - Observations and insights

---

## How to Run the Scripts

1. **Environment Setup**:
   - Install the required Python libraries:
     ```bash
     pip install numpy pandas matplotlib scikit-learn xgboost nltk
     ```
   - Ensure the `housing.csv`, `titanic.csv`, or other required datasets are placed in the project directory.

2. **Run Individual Scripts**:
   - Task 1 (EDA): Open `task1_eda.ipynb` in Jupyter Notebook and execute all cells.
   - Task 2 (Sentiment Analysis): Run `task2_sentiment_analysis.py`:
     ```bash
     python task2_sentiment_analysis.py
     ```
   - Task 3 (Fraud Detection): Run `task3_fraud_detection.py`:
     ```bash
     python task3_fraud_detection.py
     ```
   - Task 4 (House Prices): Run `task4_house_prices.py`:
     ```bash
     python task4_house_prices.py
     ```

3. **View Results**:
   - All visualizations, metrics, and model outputs are displayed in the scripts or notebooks.

---

## Observations

- Task 1: The EDA provided key insights into the dataset, revealing trends and correlations among features.
- Task 2: The sentiment analysis model achieved competitive precision, recall, and F1-scores on the IMDB Reviews dataset.
- Task 3: The fraud detection system accurately identified fraudulent transactions, with a user-friendly testing interface.
- Task 4: The custom regression models showed comparable performance, with tree-based models highlighting feature importance effectively.

---

## Submission Deadline
All tasks must be completed and submitted by **16th January 2024**.
