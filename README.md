# Predictive Trends & Sentiment ML

A comprehensive repository containing machine learning, statistical modeling, and natural language processing (NLP) workflows.

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

## Overview
This repository covers three core analytical areas: real estate price estimation, movie review sentiment classification, and macroeconomic time-series forecasting.

---

## Dependencies
The project utilizes the following core Python libraries:
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Statistical Modeling:** `statsmodels` (OLS, seasonal decomposition, Holt-Winters, AutoReg)]
* **Machine Learning:** `scikit-learn` (Linear Regression, Random Forest, Decision Trees, Neural Networks, preprocessing)
* **Time Series Forecasting:** `skforecast`
* **Natural Language Processing:** `nltk` (VADER Sentiment Intensity Analyzer)
* **Model Interpretability:** `shap`

---

## Project Tasks

### Task 1: Property Price Prediction
* **Dataset:** `Property.csv` featuring 1,750 properties across `Price`, `Area`, and `Rooms`
* **Linear Regression:** Fitted a linear model of `Price` on `Rooms` and `Area`. Achieved an $R^2$ of $0.677$, with both features showing strong statistical significance ($p < 0.001$).
* **Dummy Variables:** Converted `Rooms` into binary dummy variables[cite: 1]. This improved the model's $R^2$ to $0.703$, effectively capturing non-linear room effects.
* **Predictions:** Generated forecasts for specific property configurations (`House_1`: Area=1000, Rooms=3; `House_2`: Area=2000, Rooms=2).
* **Decision Tree:** Fitted a Decision Tree Regressor (`max_depth=4`) using the dummy-variable specification.
* **Model Comparison:** Evaluated models using an 80/20 train/test split. Linear regression demonstrated superior out-of-sample generalization (Test RMSE: 39.50) compared to the decision tree (Test RMSE: 42.66), which suffered from minor in-sample overfitting.

### Task 2: Movie Review Sentiment Analysis
* **Dataset:** `Rotten.csv` containing 900 movie reviews structured with `ID`, `Verdict`, and `Appraisal`.
* **Sentiment Extraction:** Applied VADER to extract polarity scores (`compound`, `pos`, `neg`, `neu`). Comparative bar plots confirmed higher positive and compound scores for recommended reviews (`Verdict = 1`).
* **Extreme Case Analysis:** Investigated the individual reviews yielding the highest and lowest compound sentiment scores.
* **Random Forest:** Trained a Random Forest Classifier ($60$ trees) on a 90/10 train/test split using `pos` and `neg` sentiment features, achieving an F1 score of $0.7089$.
* **Neural Network:** Fitted an MLP Classifier (1 hidden layer, 30 neurons, logistic activation). Evaluated feature impact using SHAP summary plots (`plot_type='dot'`).
* **Model Comparison:** Compared performance metrics; the neural network achieved higher recall ($0.7561$), while the random forest exhibited higher precision ($0.7368$).

### Task 3: Macroeconomic Trade Analysis
* **Dataset:** `trade.csv` containing monthly time-series records for Japanese `Exports` and `Imports`.
* **Visual Diagnostics:** Generated and analyzed time-series line plots to illustrate long-term macroeconomic trends and seasonality across the two trade series.

---

## Author
* **Sadit Khan**
## Co-author
* **Local Agent**
