# Fashion Demand Forecasting & Qualification

This project implements a data engineering and forecasting pipeline for a fashion retailer. It includes data analysis, feature engineering, product qualification logic, and demand forecasting models using LightGBM.

## Project Structure

```
mathieu_damours_assignment/
├── data/                       # Contains the input CSV dataset
├── src/                        # Source code for core logic
│   ├── models.py               # Forecasting models (Baseline, LGBM) and evaluation
│   └── product_qualification.py # Logic to determine if a product should be forecasted
├── tests/                      # Unit tests
│   └── test_product_qualification.py
├── engineering.ipynb           # Main notebook for analysis, visualization, and modeling
└── README.md                   # Project documentation
```

## Setup & Dependencies

The project requires Python 3.10+ and the following libraries:

*   `polars`: Fast DataFrame library for data manipulation.
*   `plotly`: Interactive visualization library.
*   `lightgbm`: Gradient boosting framework for forecasting.
*   `scikit-learn`: Machine learning utilities.
*   `numpy`: Numerical computing.

You can install them via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Analysis & Engineering
The notebook `engineering.ipynb` contains the complete workflow:
*   **Data Loading**: Reads the sales and inventory data.
*   **Visualization**: Plots sales, inventory, and price trends over time.
*   **Feature Engineering**: Creates features like stockout flags, lifecycle stages, and rolling metrics.
*   **Product Qualification**: Applies business rules to decide if the product is suitable for forecasting.
*   **Model Comparison**: Trains and evaluates three forecasting approaches (Baseline, LGBM with lags, LGBM with engineered features).

### 2. Running Tests
The unit tests for the product qualification logic are located in `src/unit_tests.py`.
You can run them directly from the command line:

```bash
python -m unittest tests/test_product_qualification.py
```

Or run the last cell in `engineering.ipynb`, which executes the tests within the notebook environment.

## Key Components

### Product Qualification (`src/product_qualification.py`)
Contains the `should_forecast_product` function, which evaluates a product's history against criteria such as:
*   **Inventory Availability**: Is the product in inventory?
*   **New Product**: Is it a new product?
*   **Sales Recency**: For older products, do they still sell well?
*   **Profitability**: Is the margin acceptable?

### Forecasting Models (`src/models.py`)
Implements the forecasting pipeline:
*   **Baseline**: A simple 28-day moving average.
*   **LightGBM**: A gradient boosting regressor trained on:
    *   **Lag Features**: Sales from 7, 14, and 28 days ago.
    *   **Engineered Features**: Markdown percentage, margin, weeks since launch, etc.
*   **Evaluation**: Calculates Weighted Mean Absolute Percentage Error (WMAPE) to compare model performance.

## Results
The analysis in `engineering.ipynb` demonstrates that the LightGBM model with engineered features generally outperforms the baseline and lag-only models by capturing context such as price changes and product lifecycle stages. Feature importance analysis highlights the impact of these engineered features on the forecast.
