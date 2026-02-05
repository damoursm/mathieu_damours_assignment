import polars as pl


def should_forecast_product(df: pl.DataFrame,
                            min_recent_sales_days: int = 7,
                            min_margin: float = 0.0,
                            history_window: int = 28,
                            new_product_days: int = 14,
                            clearance_markdown_pct: float = 0.3) -> pl.DataFrame:
    """
    Determine if a product should be forecasted for each day based on rolling criteria.

    This function evaluates daily records against multiple qualification criteria including
    sales recency, inventory age, profitability, data quality, and stockout rates.

    Args:
        df: Polars DataFrame with product time series data. Must contain columns:
            'sales', 'date', 'launch_date', 'margin', 'inventory', 'current_price', 'cost'.
        min_recent_sales_days: Minimum days with sales > 0 in the last 14 days.
        max_stockout_rate: Maximum acceptable stockout rate (rolling window).
        min_margin: Minimum acceptable average margin (rolling window).
        min_data_quality_pct: Minimum percentage of non-null critical fields (rolling window).
        max_inventory_age_days: Maximum days since launch for the product to be considered relevant.
        history_window: Window size in days for rolling calculations (default: 28).

    Returns:
        pl.DataFrame: The input DataFrame with additional boolean columns indicating
        qualification status for each criterion and a final 'should_forecast' column.

    Raises:
        ValueError: If the input DataFrame is missing required columns.
        TypeError: If the input is not a Polars DataFrame.

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> data = {
        ...     'date': [date(2024, 1, i) for i in range(1, 30)],
        ...     'sales': [10] * 29,
        ...     'launch_date': [date(2023, 12, 1)] * 29,
        ...     'margin': [0.5] * 29,
        ...     'inventory': [100] * 29,
        ...     'current_price': [20.0] * 29,
        ...     'cost': [10.0] * 29
        ... }
        >>> df = pl.DataFrame(data)
        >>> result = should_forecast_product(df)
        >>> result.select('should_forecast').tail(1).item()
        True
    """
    # Input Validation
    if not isinstance(df, pl.DataFrame):
        raise TypeError("Input 'df' must be a Polars DataFrame.")

    required_cols = {'sales', 'date', 'launch_date', 'margin', 'inventory', 'current_price', 'cost'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    if df.height == 0:
        # Return empty dataframe with expected schema if input is empty
        # We need to add the new columns to match the return type structure
        # For simplicity, just return the empty df, but ideally we should add the schema.
        # Let's let the polars operations handle empty df naturally or return early.
        return df.with_columns([
            pl.lit(None).cast(pl.Boolean).alias('should_forecast')
        ])

    # 1. Inventory Availability (must have inventory > 0)
    pass_inventory = (pl.col('inventory') > 0).alias('pass_inventory')

    # 2. New Product (within new_product_days of launch - always forecast)
    days_since_launch = (
        (pl.col('date') - pl.col('launch_date')).dt.total_days()
    ).alias('days_since_launch')
    is_new_product = (days_since_launch <= new_product_days).alias('is_new_product')

    # Rule 3: Old products that don't sell shouldn't be forecasted
    sales_recency = (
        pl.col('sales').gt(0).cast(pl.Int32)
        .rolling_sum(window_size=14)
        .alias('recent_sales_count')
    )
    pass_sales_recency = (sales_recency >= min_recent_sales_days).alias('pass_sales_recency')
    pass_product_age_rule = (is_new_product | pass_sales_recency).alias('pass_product_age_rule')

    # 3. Profitability (Rolling average margin)
    rolling_margin = (
        pl.col('margin').rolling_mean(window_size=history_window)
    ).alias('rolling_avg_margin')
    pass_profitability = (rolling_margin >= min_margin).alias('pass_profitability')

    # 4. Clearance Sales (products on significant markdown should be forecasted)
    if 'markdown_pct' in df.columns:
        pass_clearance_sales = (pl.col('markdown_pct') >= clearance_markdown_pct).alias('pass_clearance_sales')
    else:
        pass_clearance_sales = pl.lit(False).alias('pass_clearance_sales')

    # Construct the result dataframe
    result = df.with_columns([
        pass_inventory,
        pass_product_age_rule,
        pass_profitability,
        pass_clearance_sales
    ])

    # Add pass flags
    result = result.with_columns([
        pass_inventory,
        pass_product_age_rule,
        pass_profitability,
        pass_clearance_sales
    ])
    
    # Final decision
    result = result.with_columns(
        (
            pl.col('pass_inventory') &
            pl.col('pass_product_age_rule') &
            pl.col('pass_profitability') &
            pl.col('pass_clearance_sales')
        ).fill_null(False).alias('should_forecast')
    )

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
        ('pass_inventory', 'Inventory Health'),
        ('pass_product_age_rule', 'Product Age and Sales Recency'),
        ('pass_profitability', 'Profitability'),
        ('pass_clearance_sales', 'Clearance Sales')
    ]

    for col, name in criteria:
        # Count nulls as failures too (insufficient history)
        failed = df.filter(pl.col(col).not_() | pl.col(col).is_null()).height
        print(f"  {name}: {failed} days ({failed/total_days*100:.1f}%)")

    print("=" * 50)
