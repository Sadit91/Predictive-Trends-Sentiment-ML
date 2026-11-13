# Predictive Trends & Sentiment ML[cite: 1]

A comprehensive repository containing machine learning, statistical modeling, and natural language processing (NLP) workflows[cite: 1].

---

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Project Tasks](#project-tasks)
  - [Task 1: Property Price Prediction](#task-1-property-price-prediction)
  - [Task 2: Movie Review Sentiment Analysis](#task-2-movie-review-sentiment-analysis)
  - [Task 3: Macroeconomic Trade Analysis](#task-3-macroeconomic-trade-analysis)
- [Author](#author)

---

## Overview[cite: 1]
This repository covers three core analytical areas: real estate price estimation, movie review sentiment classification, and macroeconomic time-series forecasting[cite: 1].

---

## Dependencies[cite: 1]
The project utilizes the following core Python libraries[cite: 1]:
* **Data Manipulation & Analysis:** `pandas`, `numpy`[cite: 1]
* **Visualization:** `matplotlib`, `seaborn`[cite: 1]
* **Statistical Modeling:** `statsmodels` (OLS, seasonal decomposition, Holt-Winters, AutoReg)[cite: 1]
* **Machine Learning:** `scikit-learn` (Linear Regression, Random Forest, Decision Trees, Neural Networks, preprocessing)[cite: 1]
* **Time Series Forecasting:** `skforecast`[cite: 1]
* **Natural Language Processing:** `nltk` (VADER Sentiment Intensity Analyzer)[cite: 1]
* **Model Interpretability:** `shap`[cite: 1]

---

## Project Tasks

### Task 1: Property Price Prediction
* **Dataset:** `Property.csv` featuring 1,750 properties across `Price`, `Area`, and `Rooms`[cite: 1].
* **Linear Regression:** Fitted a linear model of `Price` on `Rooms` and `Area`[cite: 1]. Achieved an $R^2$ of $0.677$, with both features showing strong statistical significance ($p < 0.001$)[cite: 1].
* **Dummy Variables:** Converted `Rooms` into binary dummy variables[cite: 1]. This improved the model's $R^2$ to $0.703$, effectively capturing non-linear room effects[cite: 1].
* **Predictions:** Generated forecasts for specific property configurations (`House_1`: Area=1000, Rooms=3; `House_2`: Area=2000, Rooms=2)[cite: 1].
* **Decision Tree:** Fitted a Decision Tree Regressor (`max_depth=4`) using the dummy-variable specification[cite: 1].
* **Model Comparison:** Evaluated models using an 80/20 train/test split[cite: 1]. Linear regression demonstrated superior out-of-sample generalization (Test RMSE: 39.50) compared to the decision tree (Test RMSE: 42.66), which suffered from minor in-sample overfitting[cite: 1].

### Task 2: Movie Review Sentiment Analysis
* **Dataset:** `Rotten.csv` containing 900 movie reviews structured with `ID`, `Verdict`, and `Appraisal`[cite: 1].
* **Sentiment Extraction:** Applied VADER to extract polarity scores (`compound`, `pos`, `neg`, `neu`)[cite: 1]. Comparative bar plots confirmed higher positive and compound scores for recommended reviews (`Verdict = 1`)[cite: 1].
* **Extreme Case Analysis:** Investigated the individual reviews yielding the highest and lowest compound sentiment scores[cite: 1].
* **Random Forest:** Trained a Random Forest Classifier ($60$ trees) on a 90/10 train/test split using `pos` and `neg` sentiment features, achieving an F1 score of $0.7089$[cite: 1].
* **Neural Network:** Fitted an MLP Classifier (1 hidden layer, 30 neurons, logistic activation)[cite: 1]. Evaluated feature impact using SHAP summary plots (`plot_type='dot'`)[cite: 1].
* **Model Comparison:** Compared performance metrics; the neural network achieved higher recall ($0.7561$), while the random forest exhibited higher precision ($0.7368$)[cite: 1].

### Task 3: Macroeconomic Trade Analysis
* **Dataset:** `trade.csv` containing monthly time-series records for Japanese `Exports` and `Imports`[cite: 1].
* **Visual Diagnostics:** Generated and analyzed time-series line plots to illustrate long-term macroeconomic trends and seasonality across the two trade series[cite: 1].

---

## Author
* **Sadit Khan**[cite: 1]
