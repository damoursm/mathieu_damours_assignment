import polars as pl


def should_forecast_product(df: pl.DataFrame,
                            min_recent_sales_days: int = 7,
                            max_stockout_rate: float = 0.5,
                            min_margin: float = 0.0,
                            min_data_quality_pct: float = 0.9,
                            max_inventory_age_days: int = 180,
                            history_window: int = 28) -> pl.DataFrame:
    """
    Determine if a product should be forecasted for each day based on rolling criteria.

    Args:
        df: Polars DataFrame with product time series data
        min_recent_sales_days: Minimum days with sales in last 14 days
        max_stockout_rate: Maximum acceptable stockout rate (rolling)
        min_margin: Minimum acceptable average margin (rolling)
        min_data_quality_pct: Minimum percentage of non-null critical fields (rolling)
        max_inventory_age_days: Maximum days since launch to be relevant
        history_window: Window size for rolling calculations (days)

    Returns:
        DataFrame with added columns indicating qualification status and reasons.
    """
    
    # 1. Sales Recency (Last 14 days)
    # Count days with sales > 0 in last 14 days
    sales_recency = (
        pl.col('sales').gt(0).cast(pl.Int32)
        .rolling_sum(window_size=14)
        .alias('recent_sales_count')
    )
    
    pass_sales_recency = (sales_recency >= min_recent_sales_days).alias('pass_sales_recency')

    # 2. Inventory Age
    # Days since launch
    days_since_launch = (
        (pl.col('date') - pl.col('launch_date')).dt.total_days()
    ).alias('days_since_launch')
    
    pass_inventory_age = (days_since_launch <= max_inventory_age_days).alias('pass_inventory_age')

    # 3. Profitability (Rolling average margin)
    rolling_margin = (
        pl.col('margin').rolling_mean(window_size=history_window)
    ).alias('rolling_avg_margin')
    
    pass_profitability = (rolling_margin >= min_margin).alias('pass_profitability')

    # 4. Data Quality (Rolling non-null rate)
    critical_cols = ['sales', 'inventory', 'current_price', 'cost']
    
    quality_exprs = [
        pl.col(c).is_not_null().cast(pl.Float64).rolling_mean(window_size=history_window)
        for c in critical_cols
    ]
    
    # 5. Stockout Rate (Rolling)
    # If is_stockout is missing, we can't calculate it properly, but we'll try to use it if present
    if 'is_stockout' in df.columns:
        rolling_stockout = (
            pl.col('is_stockout').cast(pl.Float64).rolling_mean(window_size=history_window)
        ).alias('rolling_stockout_rate')
    else:
        # Fallback if column missing (though it should be there from engineering step)
        rolling_stockout = pl.lit(0.0).alias('rolling_stockout_rate')
    
    pass_stockout = (rolling_stockout <= max_stockout_rate).alias('pass_stockout_rate')

    # Construct the result dataframe
    result = df.with_columns([
        sales_recency,
        days_since_launch,
        rolling_margin,
        rolling_stockout
    ])
    
    # Add quality columns temporarily to calculate min
    quality_cols = [f'quality_{c}' for c in critical_cols]
    result = result.with_columns([
        expr.alias(name) for expr, name in zip(quality_exprs, quality_cols)
    ])
    
    # Calculate min quality
    result = result.with_columns(
        pl.min_horizontal(quality_cols).alias('min_data_quality')
    )
    
    pass_quality = (pl.col('min_data_quality') >= min_data_quality_pct).alias('pass_data_quality')
    
    # Add pass flags
    result = result.with_columns([
        pass_sales_recency,
        pass_inventory_age,
        pass_profitability,
        pass_stockout,
        pass_quality
    ])
    
    # Final decision
    result = result.with_columns(
        (
            pl.col('pass_sales_recency') &
            pl.col('pass_inventory_age') &
            pl.col('pass_profitability') &
            pl.col('pass_stockout_rate') &
            pl.col('pass_data_quality')
        ).fill_null(False).alias('should_forecast')
    )
    
    # Clean up temporary columns
    result = result.drop(quality_cols)
    
    return result


def print_qualification_report(df: pl.DataFrame):
    """Print a summary of the qualification results."""
    print("=" * 50)
    print("PRODUCT QUALIFICATION REPORT (Daily)")
    print("=" * 50)
    
    total_days = df.height
    qualified_days = df.filter(pl.col('should_forecast')).height
    
    print(f"Total Days: {total_days}")
    print(f"Qualified Days: {qualified_days} ({qualified_days/total_days*100:.1f}%)")
    
    print("\nFailure Reasons (Days Failed):")
    criteria = [
        ('pass_sales_recency', 'Sales Recency'),
        ('pass_inventory_age', 'Inventory Age'),
        ('pass_profitability', 'Profitability'),
        ('pass_stockout_rate', 'Stockout Rate'),
        ('pass_data_quality', 'Data Quality')
    ]

    for col, name in criteria:
        # Count nulls as failures too (insufficient history)
        failed = df.filter(pl.col(col).not_() | pl.col(col).is_null()).height
        print(f"  {name}: {failed} days ({failed/total_days*100:.1f}%)")

    print("=" * 50)
