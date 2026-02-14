import unittest
import pandas as pd
from ingest_data import fix_price
from val_model import get_nums

class TestCleaning(unittest.TestCase):

    def test_price_cleaner(self):
        # check simple price
        self.assertEqual(fix_price("$400.00"), 400.0)
        # check range price
        self.assertEqual(fix_price("$300 to $400"), 300.0)
        # check garbage
        self.assertIsNone(fix_price("Call for price"))

    def test_ram_extractor(self):
        # check standard gb
        self.assertEqual(get_nums("16 GB"), 16)
        # check weird text
        self.assertEqual(get_nums("Ram: 8GB"), 8)
        # check empty
        self.assertEqual(get_nums(None), 0)

if __name__ == '__main__':
    unittest.main()