from blocklearning.aggregators import BasilAggregator
from blocklearning.models import SimpleMLP
import tensorflow as tf
import unittest
from unittest.mock import Mock
import os

class Testing(unittest.TestCase):
    def test_calc_F1(self):
        data_tf = tf.data.experimental.load('./files/data_tf')
        model = tf.keras.models.load_model('./files/model.h5')
        weights_loader = Mock()
        basil_aggregator = BasilAggregator(weights_loader)
        f1, f1_detailed = basil_aggregator._BasilAggregator__calc_F1(data_tf, model)
        self.assertGreater(f1, 0)

if __name__ == '__main__':
    unittest.main()
