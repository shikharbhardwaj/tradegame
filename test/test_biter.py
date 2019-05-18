""" biter_test.py

Tests for the data batch iterator.
"""

import unittest
import sys
from os import path

SRC_DIR = path.join(path.dirname(__file__), '..', 'src')
sys.path.append(SRC_DIR)

from environment.data import BatchIterator

class BiterTest(unittest.TestCase):
    def test_init(self):
        """ Test batch iterator initialisation.

        Arguments:
            unittest {} --
        """
        # Test parameters
        SCHEMA = ['tick','open','high','low','close','volume']
        DATA_LOCATION = path.join(path.dirname(__file__), 'data')
        PAIRS = ['constant', 'random']
        BEGIN_YEAR = 2012
        END_YEAR = 2013

        biter = BatchIterator(DATA_LOCATION, PAIRS, BEGIN_YEAR, END_YEAR, False, 10)

        self.assertEqual(biter.location, DATA_LOCATION)
        self.assertEqual(biter.currency_pairs, PAIRS)
        self.assertEqual(biter.begin_year, BEGIN_YEAR)
        self.assertEqual(biter.end_year, END_YEAR)
        self.assertEqual(biter.period, 10)
        self.assertEqual(biter.preprocess, False)
        self.assertEqual(biter.offset, 0)

    def test_tick_progress(self):


if __name__ == '__main__':
    unittest.main()