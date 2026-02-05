import unittest
import polars as pl
from datetime import date, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.product_qualification import should_forecast_product


class TestProductQualification(unittest.TestCase):

    def setUp(self):
        """Create a base valid dataframe."""
        self.dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        self.base_data = {
            'date': self.dates,
            'sales': [10] * 50,
            'launch_date': [date(2023, 12, 1)] * 50,
            'margin': [0.5] * 50,
            'inventory': [100] * 50,
            'current_price': [20.0] * 50,
            'cost': [10.0] * 50,
        }

    def test_rule1_no_inventory_fails(self):
        """Rule 1: Products with no inventory should not be forecasted."""
        data = self.base_data.copy()
        data['inventory'] = [0] * 50
        df = pl.DataFrame(data)

        result = should_forecast_product(df)

        self.assertFalse(result.tail(1)['pass_inventory'].item())
        self.assertFalse(result.tail(1)['should_forecast'].item())

    def test_pass_profitability(self):
        """Rule: Products with sufficient margin should pass profitability."""
        data = self.base_data.copy()
        data['margin'] = [0.5] * 50  # Good margin
        df = pl.DataFrame(data)

        result = should_forecast_product(df, min_margin=0.2)

        self.assertTrue(result.tail(1)['pass_profitability'].item())

    def test_rule4_clearance_forecasted(self):
        """Rule 4: Products on clearance should be forecasted."""
        data = self.base_data.copy()
        data['markdown_pct'] = [0.5] * 50  # 50% markdown
        data['launch_date'] = [date(2020, 1, 1)] * 50  # Old product
        data['sales'] = [0] * 50  # No sales
        df = pl.DataFrame(data)

        result = should_forecast_product(df, clearance_markdown_pct=0.3)

        self.assertTrue(result.tail(1)['pass_clearance_sales'].item())


if __name__ == '__main__':
    unittest.main()