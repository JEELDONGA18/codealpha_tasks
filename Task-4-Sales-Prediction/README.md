# Sales Prediction using Machine Learning

**CodeAlpha Data Science Internship — Task 4**

A complete machine learning project that predicts product sales based on advertising expenditure across TV, Radio, and Newspaper channels. This project demonstrates the end-to-end data science workflow from data exploration to actionable business insights.

---

## Project Overview

Companies invest significant budgets into advertising, but understanding which channels actually drive sales is often based on guesswork. This project uses machine learning to analyze the relationship between advertising spend and sales performance, enabling data-driven marketing decisions.

We train multiple regression models on historical advertising data and compare their performance to find the most accurate sales predictor. The analysis reveals which advertising channels deliver the highest ROI and how businesses should allocate their marketing budgets.

## Objectives

- Understand the relationship between advertising spend and product sales
- Perform thorough exploratory data analysis and feature engineering
- Build, train, and compare multiple regression models
- Evaluate model performance using standard metrics (MAE, MSE, RMSE, R²)
- Identify the most influential advertising channels
- Generate actionable marketing insights for business decision-making

## Dataset Information

| Feature | Description |
|---------|-------------|
| TV | TV advertising budget (in $1000s) |
| Radio | Radio advertising budget (in $1000s) |
| Newspaper | Newspaper advertising budget (in $1000s) |
| Sales | Product sales (in 1000 units) — **Target Variable** |

- **Source**: Advertising dataset
- **Samples**: 200
- **Features**: 3 (TV, Radio, Newspaper)
- **Target**: Sales

## Workflow

```
Data Collection → Data Cleaning → EDA → Feature Engineering → Visualization
    → Model Training → Model Evaluation → Feature Importance → Prediction → Insights
```

1. **Data Understanding** — Explore dataset structure, types, and statistics
2. **Data Cleaning** — Handle missing values, duplicates, and data types
3. **Exploratory Data Analysis** — Correlation, distribution, and outlier analysis
4. **Feature Engineering** — Create total budget, channel shares, interaction features
5. **Visualization** — 13+ professional plots revealing data patterns
6. **Model Training** — Train 4 regression models
7. **Model Evaluation** — Compare using MAE, MSE, RMSE, R² Score
8. **Feature Importance** — Identify which channels drive sales
9. **Sales Prediction** — Predict sales for new advertising scenarios
10. **Marketing Insights** — Actionable recommendations for marketing teams

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models |
| Joblib | Model serialization |
| Jupyter Notebook | Interactive development |

## Machine Learning Models

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Decision Tree Regressor | Non-linear tree-based model |
| Random Forest Regressor | Ensemble of decision trees |
| Gradient Boosting Regressor | Sequential ensemble method |

## Evaluation Metrics

- **MAE (Mean Absolute Error)** — Average prediction error in absolute terms
- **MSE (Mean Squared Error)** — Average squared error, penalizes large errors
- **RMSE (Root Mean Squared Error)** — Square root of MSE, same units as sales
- **R² Score** — Proportion of variance explained by the model (1.0 = perfect)

## Results

The models were compared on a 20% hold-out test set. Key findings:

- **TV advertising** is the strongest sales driver with the highest feature importance
- **Radio advertising** adds significant value, especially in combination with TV
- **Newspaper advertising** has minimal impact on sales performance
- The best model accurately predicts sales with a high R² Score

> Run the notebook to see the exact performance numbers for each model.

## Visualizations

The project generates 13+ professional visualizations including:

- Sales Distribution
- Advertising Spend Distribution (TV, Radio, Newspaper)
- TV vs Sales (with regression line)
- Radio vs Sales (with regression line)
- Newspaper vs Sales (with regression line)
- Correlation Heatmap
- Pairplot
- Boxplots for Outlier Detection
- Sales by Budget Category
- Channel Contribution (Pie Chart)
- Model Performance Comparison
- Feature Importance
- Actual vs Predicted Sales
- Residual Plot

All figures are saved to `reports/figures/`.

## Folder Structure

```
Task-4-Sales-Prediction/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Advertising.csv
│   └── processed/                    # Cleaned dataset
│       └── advertising_cleaned.csv
│
├── notebooks/
│   └── sales_prediction.ipynb        # Main analysis notebook
│
├── src/
│   ├── data_preprocessing.py         # Data loading and cleaning
│   ├── train_model.py                # Model training and selection
│   ├── evaluate_model.py             # Evaluation and visualization
│   └── predict.py                    # Sales prediction CLI
│
├── models/
│   └── trained_model.pkl             # Saved best model
│
├── reports/
│   ├── figures/                      # Saved visualizations
│   └── summary.md                    # Project summary
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

## Future Improvements

- Add more features like seasonality, campaign duration, and demographics
- Collect a larger, more diverse dataset
- Experiment with time-series forecasting for sales trends
- Implement cross-validation for more robust model evaluation
- Build an interactive web dashboard for marketing teams
- Deploy the model as a REST API for real-time predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Task-4-Sales-Prediction.git
cd Task-4-Sales-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/sales_prediction.ipynb
```
Run all cells from top to bottom to see the complete analysis.

### Command Line Prediction
```bash
cd src
python predict.py
```
Enter advertising budgets when prompted to get sales predictions.

### Run Source Modules
```bash
cd src
python data_preprocessing.py    # Test data loading
python train_model.py           # Train and save model
python evaluate_model.py        # Evaluate all models
```

## Author

**CodeAlpha Data Science Intern**

This project was built as part of the CodeAlpha Data Science Internship (Task 4 — Sales Prediction using Python).

---

*If you found this project helpful, please consider giving it a ⭐ on GitHub!*
