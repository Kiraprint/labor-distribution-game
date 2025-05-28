import unittest
from src.utils.shapley import calculate_shapley_value
from src.utils.owen import calculate_owen_value

class TestUtils(unittest.TestCase):

    def test_calculate_shapley_value(self):
        # Test case for Shapley value calculation
        coalition_values = {frozenset(['A']): 10, frozenset(['B']): 20, frozenset(['A', 'B']): 30}
        expected_value_A = 15.0  # Expected Shapley value for agent A
        expected_value_B = 25.0  # Expected Shapley value for agent B
        self.assertAlmostEqual(calculate_shapley_value('A', coalition_values), expected_value_A)
        self.assertAlmostEqual(calculate_shapley_value('B', coalition_values), expected_value_B)

    def test_calculate_owen_value(self):
        # Test case for Owen value calculation
        coalition_values = {frozenset(['A']): 10, frozenset(['B']): 20, frozenset(['A', 'B']): 30}
        expected_value_A = 15.0  # Expected Owen value for agent A
        expected_value_B = 25.0  # Expected Owen value for agent B
        self.assertAlmostEqual(calculate_owen_value('A', coalition_values), expected_value_A)
        self.assertAlmostEqual(calculate_owen_value('B', coalition_values), expected_value_B)

if __name__ == '__main__':
    unittest.main()