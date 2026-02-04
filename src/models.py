import polars as pl
import lightgbm as lgb
import numpy as np

def split_data(df: pl.DataFrame, train_days: int):
    """Splits data into training and testing sets."""
    train_df = df.head(train_days)
    test_df = df.tail(df.height - train_days)
    return train_df, test_df

def baseline_moving_average(df: pl.DataFrame, window_size: int = 28):
    """Calculates a simple moving average as a baseline forecast."""
    return df.with_columns(
        pl.col('sales').rolling_mean(window_size=window_size).shift(1).alias('forecast')
    )

def create_lag_features(df: pl.DataFrame, lags: list[int]):
    """Creates lag features for the sales column."""
    return df.with_columns(
        [pl.col('sales').shift(lag).alias(f'sales_lag_{lag}') for lag in lags]
    )

def train_lgbm(train_df: pl.DataFrame, test_df: pl.DataFrame, features: list[str]):
    """Trains a LightGBM model and returns predictions and the trained model."""
    
    # Filter out rows with nulls in features (due to lags)
    train_ready = train_df.select(['sales'] + features).drop_nulls()

    X_train = train_ready.select(features).to_numpy()
    y_train = train_ready.select('sales').to_numpy().ravel()
    
    X_test = test_df.select(features).to_numpy()

    model = lgb.LGBMRegressor(random_state=42)
    # Pass feature names explicitly to ensure they are tracked
    model.fit(X_train, y_train, feature_name=features)
    
    predictions = model.predict(X_test)
    
    return test_df.with_columns(pl.Series(name='forecast', values=predictions)), model

def calculate_wmape(df: pl.DataFrame):
    """Calculates the Weighted Mean Absolute Percentage Error."""
    # Ensure we don't divide by zero
    valid_sales = df.filter(pl.col('sales') > 0)
    
    if valid_sales.height == 0:
        return None

    numerator = np.abs(valid_sales['sales'] - valid_sales['forecast']).sum()
    denominator = valid_sales['sales'].sum()
    
    return numerator / denominator if denominator > 0 else 0.0
