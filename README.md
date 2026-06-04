# Housing Prices Prediction Using Machine Learning

A machine learning project that predicts residential house sale prices in Ames, Iowa using various regression models. Based on the [Kaggle Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course) dataset.

## Dataset

The dataset contains 81 features describing residential homes in Ames, Iowa (2006-2010), including:

- **Structural**: square footage, number of rooms/bathrooms, garage size, basement area
- **Quality**: overall quality/condition ratings, exterior/interior material quality
- **Location**: neighborhood, proximity to roads/railroads
- **Other**: lot size, year built, sale conditions

| File | Description |
|------|-------------|
| `train.csv` | 1,460 houses with sale prices (target variable) |
| `test.csv` | 1,459 houses for prediction |
| `sample_submission.csv` | Example submission format |

## Project Structure

```
.
├── Main.ipynb                  # Primary notebook - full pipeline
├── housing-price.ipynb         # Exploratory notebook - outlier analysis & feature engineering
├── home-data-for-ml-course/    # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── result.csv                  # Predicted sale prices
```

## Approach

### Data Cleaning & Imputation
- Categorical missing values filled with mode or "NA" (indicating absence of feature)
- Numerical missing values filled with mean or median based on outlier percentage
- Before/after visualizations to validate imputation quality

### Feature Engineering
- **HouseAge** / **HouseRemodAge**: derived from year sold minus year built/remodeled
- **TotalSF**: combined finished square footage across all floors
- **TotalArea**: above-ground living area + basement
- **TotalBaths**: full baths + 0.5 * half baths (including basement)
- **TotalPorchSF**: combined porch square footage
- Log transformation applied to SalePrice to normalize distribution

### Encoding
- **One-Hot Encoding**: nominal categorical features (neighborhood, sale type, etc.)
- **Ordinal Encoding**: quality/condition features with natural ordering

### Models Evaluated

| Model | R² Score (CV) |
|-------|--------------|
| XGBRegressor | 0.762 |
| GradientBoostingRegressor | 0.755 |
| RandomForestRegressor | 0.734 |
| SGDRegressor | 0.727 |
| LinearRegression | 0.682 |
| KNeighborsRegressor | 0.656 |
| DecisionTreeRegressor | 0.485 |

Additional models explored with hyperparameter tuning (GridSearchCV): XGBoost, Random Forest, Ridge, Lasso, LightGBM, and CatBoost.

## Key Findings

- **OverallQual** (0.79) and **GrLivArea** (0.71) have the strongest correlation with sale price
- Negatively correlated quality features (ordinal-encoded): ExterQual, BsmtQual, HeatingQC, KitchenQual, GarageFinish
- Neighborhood significantly impacts price: NoRidge ($335k avg) vs MeadowV ($99k avg)
- Gradient boosting methods (XGBoost, GBR) consistently outperform other approaches

## Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- lightgbm, catboost (used in exploratory notebook)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

## Usage

1. Open `Main.ipynb` in Jupyter Notebook or VS Code
2. Run all cells to execute the full pipeline: data loading, cleaning, encoding, model training, and prediction
3. Results are saved to `result.csv`
