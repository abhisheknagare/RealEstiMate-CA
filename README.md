# RealEstiMate-CA

## California Property Close Price Prediction

## Project Overview

This project aims to develop a machine learning model to estimate the **final sales price (close price)** of residential single-family properties in California. The model will use only features that are available to an end user interested in estimating the current value of any property, regardless of whether it is for sale. Our solution is akin to building our own version of Zillow’s Zestimate, focusing strictly on features that a consumer could reasonably provide or look up.

---

## Data Access and Preparation

### Data Source

- **Provider:** California Regional Multiple Listing Service (CRMLS)
- **Data Location:** FTP server provided by idxexchange.com
    - Host: `ftp.boxgrad.com`
    - Username: `data@idxexchange.com`
    - Port: `21`
- **Relevant Files:** In the `/raw/California` folder with prefix `CRMLSSold`
- **Meta Data Documentation:** `Trestle Property MetaData.pdf` in the `resources` folder

### Download Instructions

1. Download FileZilla Client from [https://filezilla-project.org](https://filezilla-project.org).
2. Connect to the FTP using the credentials above.
3. Navigate to `raw/California` and **copy** (do not modify) the latest 6 months of `CRMLSSold` files to your local machine.
4. Refer to `Trestle Property MetaData.pdf` for schema and feature details.

---

## Data Requirements

- **Filter only for:**
    - `PropertyType = "Residential"`
    - `PropertySubType = "SingleFamilyResidence"`
- **Exclude:**
    - Any features or columns that would not be available for a property currently *not* for sale (e.g., `ListPrice`, MLS listing-only fields).
    - Fields with all missing values.

---

## Task Specification

### 1. Data Exploration
- Inspect data structure and feature availability.
- Identify and select only those features that an end user could provide or research, such as:
    - Square footage (living area)
    - Lot size
    - Address (city, zip code, etc.)
    - Number of bedrooms/bathrooms
    - Year built
    - Other public property attributes
- Optionally, consider merging external data sources (e.g., local interest rates, economic indicators) if they can improve predictions and are accessible to users.

### 2. Data Preprocessing
- Handle missing values appropriately.
- Encode categorical features as needed.
- Scale numerical features if necessary.
- Split data into training and testing sets (using the latest 3–6 months of data).

### 3. Model Selection
- Start with simple models (e.g., linear regression).
- Experiment with advanced models (decision trees, random forests, gradient boosting, etc.).
- Justify model selection choices.

### 4. Model Training
- Train selected models using training data.
- Tune hyperparameters for optimal performance.

### 5. Model Evaluation
- Evaluate using appropriate metrics (e.g., R-squared) on the test set.
- Document model performance and areas for improvement.

### 6. Prediction
- Use the trained model to predict the close price for any property given user-supplied features.
- Ensure the model can be integrated into a web application, with required inputs matching those a user can provide.

### 7. Documentation
- Document all steps: exploration, preprocessing, modeling, evaluation, and prediction.
- Provide clear reasoning for all decisions.
- Prepare a live presentation summarizing findings, model performance, and the prediction process.

---

## Deliverables

- **Python Script:** End-to-end code for preprocessing, training, evaluation, and prediction.
- **Documentation:** Detailed write-up of the workflow, findings, and rationale.
- **Presentation:** Slides and/or talking points for a live Zoom presentation to stakeholders.

---

## Project Goal

To build a robust, user-focused machine learning model that accurately predicts the close price of any single-family residential property in California using only features available to consumers—enabling integration into a web application for real-time property value estimation.
