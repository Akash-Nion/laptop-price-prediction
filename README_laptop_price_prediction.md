# 💻 Laptop Price Prediction Using Machine Learning

## 📌 Project Overview

This project aims to predict laptop prices using machine learning
techniques. By analyzing key hardware specifications and brand-related
features, the model learns patterns in the data and estimates laptop
prices accurately.

The project demonstrates the complete machine learning workflow: - Data
preprocessing - Feature engineering - Model training - Model
evaluation - Result interpretation

------------------------------------------------------------------------

## 📊 Dataset

-   **File location:** `data/`
-   **Dataset name:** `Cleaned_Laptop_data.csv`
-   The dataset contains features such as:
    -   Brand
    -   Processor type
    -   RAM
    -   Storage
    -   GPU
    -   Operating system
    -   Screen size
-   **Target Variable:** `Price`

------------------------------------------------------------------------

## 🧠 Methodology

1.  Load dataset from the `data` folder\
2.  Data cleaning and preprocessing\
3.  Handling categorical variables (encoding)\
4.  Feature selection and transformation\
5.  Train regression models\
6.  Evaluate model performance

------------------------------------------------------------------------

## 🤖 Algorithms Used

-   Linear Regression\
-   Other regression models (as implemented in the notebook)

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   R² Score\
-   Mean Absolute Error (MAE)\
-   Mean Squared Error (MSE)

------------------------------------------------------------------------

## 📁 Project Structure

laptop-price-prediction/ │── README.md │── data/ │ └── Cleaned_Laptop_data.csv
│── notebooks/ │ └── laptop-price-prediction.ipynb │── src/ │ └──
laptop_price_prediction.py

------------------------------------------------------------------------

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

pip install pandas numpy scikit-learn matplotlib seaborn

### 2️⃣ Run Python Script

python src/laptop_price_prediction.py

### 3️⃣ Run Notebook

Open: notebooks/laptop-price-prediction.ipynb using Jupyter Notebook or
Google Colab.

------------------------------------------------------------------------

## 🎯 Results Summary

The trained regression model successfully identifies relationships
between laptop specifications and their prices, demonstrating the
effectiveness of machine learning in price prediction tasks.

------------------------------------------------------------------------

## 👤 Author

Nion Rahaman Akash\
Background in Mathematics & Data Science
