from blocklearning.models import SimpleMLP
import tensorflow as tf
import unittest
from unittest.mock import Mock
import os

class Testing(unittest.TestCase):
    def test_creation(self):
        model = SimpleMLP.build(784, 10)

        self.assertTrue(model != None)

if __name__ == '__main__':
    unittest.main()
