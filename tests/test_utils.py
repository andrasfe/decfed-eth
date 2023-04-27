from blocklearning.models import SimpleMLP
from blocklearning.aggregators import utils
import tensorflow as tf
import unittest
from unittest.mock import Mock
import os
import numpy as np

def flatten_nested_list(nested_list):
    flat_list = []
    for elem in nested_list:
        if isinstance(elem, list):
            flat_list.extend(flatten_nested_list(elem))
        else:
            flat_list.append(elem)
    return flat_list

class Testing(unittest.TestCase):
    def test_flatten_nested_tensor(self):
        weights = [SimpleMLP.build_mnist().get_weights() for i in range(10)]
        trainers = [str(i) for i in range(10)]

        result = utils.score(weights, trainers)
        self.assertTrue(result != None)

    # def test_flatten_nested_tensor(self):
    #     nested_list = [[1, 2, [3, 4, [5, 6]]], [7, [8]], 9]
    #     flat_list = flatten_nested_list(nested_list)
    #     print(flat_list)
    

if __name__ == '__main__':
    unittest.main()