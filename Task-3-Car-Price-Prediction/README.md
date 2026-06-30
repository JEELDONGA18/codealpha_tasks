# Car Price Prediction using Machine Learning

A machine learning project that predicts the selling price of used vehicles based on their features such as manufacturing year, present price, kilometers driven, fuel type, transmission, seller type, and ownership details.

**Built as part of the CodeAlpha Data Science Internship — Task 3**

---

## Project Overview

The resale value of a vehicle depends on multiple factors, including its age, condition, fuel type, and market demand. Accurately estimating a vehicle's selling price helps buyers, sellers, and dealerships make informed decisions.

In this project, I performed exploratory data analysis, feature engineering, and built multiple regression models to predict vehicle selling prices. The project follows the complete machine learning workflow and compares different regression algorithms to identify the best-performing model.

---

## Objectives

- Analyze factors affecting vehicle selling prices
- Explore relationships between vehicle features and resale value
- Perform data cleaning and feature engineering
- Build and compare multiple regression models
- Predict vehicle selling prices with high accuracy
- Generate business insights for buyers, sellers, and dealerships

---

## Dataset

The dataset contains information about used vehicles and their selling prices.

| Feature | Description |
|---------|-------------|
| Car_Name | Vehicle name |
| Year | Manufacturing year |
| Selling_Price | Selling price (Target Variable) |
| Present_Price | Current showroom price |
| Driven_kms | Total kilometers driven |
| Fuel_Type | Petrol, Diesel or CNG |
| Selling_type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

**Dataset Size**

- **302 Records**
- **9 Features**
- **Target Variable:** Selling_Price

---

## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Data Visualization (12+ Charts)
5. Feature Engineering
6. Model Building (4 Regression Models)
7. Model Evaluation and Comparison
8. Feature Importance Analysis
9. Business Insights and Conclusion

---

## Technologies Used

- **Python 3.9+**
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Machine Learning
- **Joblib** — Model serialization
- **Jupyter Notebook** — Interactive development

---

## Machine Learning Models

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | *Generated after notebook execution* |
| Decision Tree Regressor | *Generated after notebook execution* |
| Random Forest Regressor | *Generated after notebook execution* |
| Gradient Boosting Regressor | *Generated after notebook execution* |

> The notebook automatically evaluates all models and selects the best-performing one based on R² Score and RMSE.

**Why these models?**

- **Linear Regression** — Baseline regression model
- **Decision Tree Regressor** — Captures non-linear relationships
- **Random Forest Regressor** — Ensemble model with improved accuracy
- **Gradient Boosting Regressor** — Boosting algorithm for higher predictive performance

---

## Key Findings

- Present showroom price is one of the strongest indicators of resale value.
- Vehicle age negatively impacts the selling price.
- Cars with fewer kilometers driven generally have higher resale value.
- Fuel type influences market value, with Diesel and Petrol vehicles showing different pricing trends.
- Dealer and Individual sellers exhibit different pricing patterns.
- Ensemble models outperform simple linear regression for this dataset.

---

## Visualizations

The project includes 12+ professional visualizations:

- Selling Price Distribution
- Vehicle Age Analysis
- Present Price vs Selling Price
- Fuel Type Comparison
- Transmission Analysis
- Seller Type Analysis
- Correlation Heatmap
- Feature Importance
- Model Comparison
- Actual vs Predicted
- Residual Plot
- And more...

All figures are available in the `reports/figures/` directory.

---

## Folder Structure

```text
Task-3-Car-Price-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── car_price_prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/
│   └── trained_model.pkl
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/Task-3-Car-Price-Prediction.git
cd Task-3-Car-Price-Prediction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```

4. Run all cells from top to bottom.

---

## Future Improvements

- Deploy the model using Streamlit.
- Experiment with XGBoost and CatBoost.
- Increase dataset size for better generalization.
- Perform hyperparameter tuning.
- Build a vehicle price prediction web application.

---

## Author

**Jeelkumar Hasmukhbhai Donga**

Data Science Intern at CodeAlpha

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.