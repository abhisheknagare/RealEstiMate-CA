# RealEstiMate-CA

## California Property Close Price Prediction

## Project Overview

This is a self-initiated data science project where I set out to build a machine learning model capable of estimating the final sales price (close price) of single-family residential properties in California.

Inspired by platforms like Zillow’s Zestimate, my goal was to develop a consumer-focused property value estimator using only publicly available or user-provided property attributes — avoiding any MLS-only or non-public features.

The project follows an end-to-end ML pipeline: from raw data acquisition and cleaning to model training, evaluation, and deployment readiness.

⸻

## Data Source & Access

The dataset was obtained from the California Regional Multiple Listing Service (CRMLS) through an FTP data feed. Only records meeting the following criteria were retained:
	•	PropertyType = "Residential"
	•	PropertySubType = "SingleFamilyResidence"

To ensure consumer usability, I excluded any fields not available for properties currently off-market (e.g., ListPrice, MLS-specific IDs).

⸻

## Data Preparation & Feature Engineering
- Filtered 50K+ property records to relevant entries.
- Selected key features such as:
- Living area (sq ft)
- Lot size
- City, Zip code
- Number of bedrooms and bathrooms
- Year built
- Geographic coordinates
- Encoded categorical variables using one-hot encoding.
- Imputed missing values with statistically appropriate methods.
- Normalized numerical features where applicable.

⸻

## Exploratory Data Analysis (EDA)
- Investigated distribution of target variable (ClosePrice).
- Plotted feature-target relationships (scatter, boxplots, correlation heatmaps).
- Identified skewness and outliers in price-related fields.

⸻

## Modeling Approach

I experimented with four different regression models to compare predictive performance:
1.	Linear Regression – Baseline model to establish a reference R² score.
2.	Random Forest Regressor – Captures non-linear relationships with ensemble learning.
3.	XGBoost Regressor – Gradient boosting model optimized for performance on structured data.
4.	HistGradientBoostingRegressor – A fast, histogram-based gradient boosting implementation.

⸻

## Model Evaluation

### Evaluation metrics:
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Key findings:
- Linear Regression achieved an R² of 79% (baseline).
- Random Forest and XGBoost significantly improved performance, with XGBoost giving the best R² of ~88%.
- HistGradientBoostingRegressor performed closely to XGBoost but with faster training times.

⸻

## Prediction & Integration Readiness

The trained models accept user-supplied features and output an estimated Close Price for any property in California meeting the criteria. This design allows for easy integration into a web app or API service for real-time property valuation.

⸻

## Future Improvements
- Incorporate external data such as interest rates, school ratings, and local economic indicators.
- Fine-tune hyperparameters for gradient boosting models to maximize accuracy.
- Build an interactive Streamlit or Flask web app for user-facing predictions.

⸻

## Tech Stack
- Languages & Libraries: Python, Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost
- Visualization: Matplotlib, Seaborn
- Models: Linear Regression, Random Forest, XGBoost, HistGradientBoostingRegressor
- Environment: Jupyter Notebook

⸻

## Outcome

This project demonstrates my ability to:
- Acquire and clean large-scale structured datasets.
- Engineer features for real-world prediction tasks.
- Compare and evaluate multiple ML algorithms.
- Prepare models for production-level integration.

## Project Goal

To build a robust, user-focused machine learning model that accurately predicts the close price of any single-family residential property in California using only features available to consumers enabling integration into a web application for real-time property value estimation.
