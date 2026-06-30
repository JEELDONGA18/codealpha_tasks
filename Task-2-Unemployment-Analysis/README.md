# Unemployment Analysis in India

A data science project analyzing unemployment trends across Indian states from 2019 to 2020, with a focus on understanding the impact of COVID-19 on the Indian labour market.

**Built as part of the CodeAlpha Data Science Internship — Task 2**

---

## Project Overview

The COVID-19 pandemic caused massive disruptions to economies worldwide, and India was no exception. This project explores how unemployment rates changed across different states and regions before, during, and after the initial lockdown period.

Using real-world data from the Centre for Monitoring Indian Economy (CMIE), I performed exploratory data analysis, created visualizations to uncover patterns, and built machine learning models to understand the key factors driving unemployment.

---

## Objectives

- Analyze unemployment trends across Indian states (May 2019 – October 2020)
- Compare unemployment between rural and urban areas
- Understand the impact of COVID-19 on employment
- Identify the most and least affected states and regions
- Build and compare regression models to predict unemployment rates

---

## Dataset

Two datasets were used in this analysis:

| Dataset | Records | Time Period | Source |
|---------|---------|-------------|--------|
| Unemployment in India | 740+ | May 2019 – June 2020 | CMIE / Kaggle |
| Unemployment Rate upto 11/2020 | 268 | Jan 2020 – October 2020 | CMIE / Kaggle |

**Key Columns:**
- Region (State name)
- Date
- Estimated Unemployment Rate (%)
- Estimated Employed
- Estimated Labour Participation Rate (%)
- Area (Rural / Urban)

---

## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Data Visualization (14+ charts)
5. Feature Engineering
6. Model Building (3 algorithms)
7. Model Evaluation and Comparison
8. Insights and Conclusion

---

## Technologies Used

- **Python 3.9+**
- **pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical plots
- **scikit-learn** — Machine learning
- **Jupyter Notebook** — Interactive development

---

## Machine Learning Models

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | 6.819892 | 102.635540 | 10.130920 | 0.291082 | 
| Decision Tree | 5.232838 | 89.397005 | 9.454999 | 0.382522 | 
| Random Forest | 4.292999 | 60.855059 | 7.800965 | 0.579666 | 

> Note: Actual scores will appear after running the notebook. Random Forest is expected to perform best.

**Why these models?**
- **Linear Regression** — Simple baseline to check if the relationship is linear
- **Decision Tree** — Captures non-linear patterns, easy to interpret
- **Random Forest** — Ensemble method that handles complex patterns and reduces overfitting

---

## Key Findings

- Unemployment spiked dramatically during April–May 2020 due to the nationwide lockdown
- Urban areas experienced higher unemployment than rural areas
- States like Bihar, Jharkhand, and Puducherry were among the hardest hit
- Northern and Eastern India saw higher unemployment compared to Southern states
- The labour market showed signs of recovery by June–July 2020, though unevenly

---

## Visualizations

The project includes 14+ professional charts:
- State-wise unemployment comparison
- Time series trends
- Rural vs Urban analysis
- COVID impact box plots
- Correlation heatmap
- Model performance comparison
- Actual vs Predicted scatter plot
- And more...

All figures are saved in the `reports/figures/` directory.

---

## Folder Structure

```
Task-2-Unemployment-Analysis/
│
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned datasets
│
├── notebooks/
│   └── unemployment_analysis.ipynb   # Main analysis notebook
│
├── src/
│   ├── data_preprocessing.py   # Data loading and cleaning
│   ├── train_model.py          # Model training functions
│   ├── evaluate_model.py       # Evaluation metrics and plots
│   └── predict.py              # Prediction utilities
│
├── models/
│   └── random_forest_model.pkl # Saved trained model
│
├── reports/
│   └── figures/                # Generated visualizations
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## How to Run

1. Clone this repository
```bash
git clone https://github.com/yourusername/Task-2-Unemployment-Analysis.git
cd Task-2-Unemployment-Analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook notebooks/unemployment_analysis.ipynb
```

4. Run all cells from top to bottom

---

## Future Improvements

- Add time series forecasting models (ARIMA, Prophet) to predict future trends
- Include more recent data (2021–2024) to analyze post-pandemic recovery
- Build an interactive dashboard using Plotly or Streamlit
- Perform deeper analysis on sector-wise unemployment
- Add geographic heatmaps using Folium

---

## Author

**Jeelkumar Hasmukhbhai Donga**

Data Science Intern at CodeAlpha

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
