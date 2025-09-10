import unittest
import torch
import numpy as np
from src.utils.data_loader import get_datasets, add_pattern_bd, poison_dataset, DatasetSplit

class TestDataLoader(unittest.TestCase):

    def test_get_datasets(self):
        """Test if datasets are loaded correctly."""
        train_dataset, test_dataset = get_datasets('mnist', data_dir='./data')
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(test_dataset)
        self.assertEqual(len(train_dataset), 60000)
        self.assertEqual(len(test_dataset), 10000)

    def test_add_pattern_bd(self):
        """Test if the backdoor pattern is added correctly."""
        # Create a dummy image
        img = torch.zeros(28, 28)
        # Add a square pattern
        bd_img = add_pattern_bd(img, dataset='mnist', pattern_type='square')
        # Check if the pattern is present
        self.assertEqual(bd_img[5, 6], 255)
        self.assertEqual(bd_img[6, 10], 255)

    def test_poison_dataset(self):
        """Test if the dataset is poisoned correctly."""
        train_dataset, _ = get_datasets('mnist', data_dir='./data')
        # Poison 10% of the images of class 5 to class 7
        base_class = 5
        target_class = 7
        poison_frac = 0.1
        
        # Get the original indices for the base class
        original_indices = (train_dataset.targets == base_class).nonzero().flatten().tolist()
        
        # Poison the dataset
        poison_dataset(train_dataset, 'mnist', base_class, target_class, poison_frac, 'square')
        
        # Get the new indices for the target class
        new_indices = (train_dataset.targets == target_class).nonzero().flatten().tolist()
        
        # Check if some of the original indices are now in the new indices
        self.assertTrue(any(idx in new_indices for idx in original_indices))

if __name__ == '__main__':
    unittest.main()
