""" env_test.py

Tests for the simulated trading environment for proper action execution and
state representation.
"""

import unittest
import sys
from os import path

SRC_DIR = path.join(path.dirname(__file__), '..', 'src')
sys.path.append(SRC_DIR)

from environment import environment

class EnvTest(unittest.TestCase):
    def test_init(self):
        """ Test environment initializtion.

        Arguments:
            unittest {} --
        """
        pairs = ['A', 'B', 'C', 'D']

if __name__ == '__main__':
    unittest.main()
