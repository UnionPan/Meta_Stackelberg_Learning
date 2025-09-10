import unittest
import numpy as np
import torch
from src.utils.fl_utils import get_parameters, set_parameters, weights_to_vector, vector_to_weights, average
from src.models.cnn import ResNet18

class TestFlUtils(unittest.TestCase):

    def test_weight_conversion(self):
        """Test the conversion between weights and vector."""
        model = ResNet18()
        weights = get_parameters(model)
        vector = weights_to_vector(weights)
        converted_weights = vector_to_weights(vector, weights)

        # Check if the shapes are the same
        for i in range(len(weights)):
            self.assertEqual(weights[i].shape, converted_weights[i].shape)

        # Check if the values are the same
        for i in range(len(weights)):
            self.assertTrue(np.allclose(weights[i], converted_weights[i]))

    def test_average(self):
        """Test the average aggregation rule."""
        weights1 = [np.ones((2, 2)), np.ones((3, 3))]
        weights2 = [np.ones((2, 2)) * 3, np.ones((3, 3)) * 3]
        
        aggregated_weights = average([weights1, weights2], num_clients=2, subsample_rate=1.0)
        
        # The average should be [[2, 2], [2, 2]] and [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.assertTrue(np.allclose(aggregated_weights[0], np.ones((2, 2)) * 2))
        self.assertTrue(np.allclose(aggregated_weights[1], np.ones((3, 3)) * 2))

    # TODO: Add more tests for other aggregation rules and attack crafting functions.

if __name__ == '__main__':
    unittest.main()
