# Sales Prediction — Project Summary

**CodeAlpha Data Science Internship — Task 4**

## Problem Statement

Predict product sales based on advertising spend across TV, Radio, and Newspaper channels using machine learning regression models.

## Dataset

- **Source**: Advertising dataset (200 samples)
- **Features**: TV, Radio, Newspaper (advertising budgets in $1000s)
- **Target**: Sales (in 1000 units)
- **Quality**: No missing values, no duplicates, clean numerical data

## Data Preprocessing

- Dropped unnamed index column
- Verified no missing values or duplicate records
- Created engineered features: Total_Budget, channel share percentages, TV_Radio_Interaction, Budget_Category
- Split data 80/20 for training and testing (random_state=42)

## Exploratory Data Analysis Findings

1. **TV** has the strongest correlation with Sales (r ≈ 0.78)
2. **Radio** shows moderate correlation (r ≈ 0.58)
3. **Newspaper** has weak correlation (r ≈ 0.23)
4. Sales range from 1.6k to 27.0k units with a mean of ~14.0k
5. Higher budget categories consistently produce higher average sales
6. Newspaper has a few outliers but they represent genuine data points

## Models Trained

| Model | Type |
|-------|------|
| Linear Regression | Baseline linear model |
| Decision Tree Regressor | Non-linear, tree-based |
| Random Forest Regressor | Ensemble of 100 trees |
| Gradient Boosting Regressor | Sequential ensemble, 100 estimators |

## Evaluation Metrics Used

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

## Key Results

- The ensemble models (Random Forest and Gradient Boosting) outperformed simpler models
- TV advertising is the dominant sales predictor by feature importance
- Radio has meaningful but secondary impact
- Newspaper contributes minimally to sales prediction

## Marketing Insights

1. **Prioritize TV advertising** — highest ROI channel for driving sales
2. **Invest in Radio** as a complementary channel, especially alongside TV
3. **Reduce Newspaper spend** — reallocate toward higher-performing channels
4. **Multi-channel strategies** (TV + Radio) outperform single-channel approaches
5. **Diminishing returns** exist at very high budget levels
6. **Data-driven budget allocation** outperforms intuition-based marketing decisions

## Visualizations Generated

13+ professional plots saved to `reports/figures/`:
- Distribution plots (Sales, TV, Radio, Newspaper)
- Scatter plots with regression lines (TV/Radio/Newspaper vs Sales)
- Correlation Heatmap
- Pairplot
- Boxplots (Outlier Detection)
- Sales by Budget Category
- Channel Contribution (Pie Chart)
- Model Comparison Charts
- Feature Importance
- Actual vs Predicted Sales
- Residual Plot

## Files Produced

- `notebooks/sales_prediction.ipynb` — Complete analysis notebook
- `src/data_preprocessing.py` — Data loading and cleaning
- `src/train_model.py` — Model training and selection
- `src/evaluate_model.py` — Evaluation and visualization
- `src/predict.py` — Prediction CLI tool
- `models/trained_model.pkl` — Saved best model
- `data/processed/advertising_cleaned.csv` — Cleaned dataset

---

*CodeAlpha Data Science Internship — Task 4*
