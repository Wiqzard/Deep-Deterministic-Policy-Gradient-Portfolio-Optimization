import unittest

class TestCalculateReward(unittest.TestCase):

    def test_calculate_reward(self):
        # Create an instance of the function to test
        calc = CalculateReward()

        # Define some input and expected output values
        inputs = [
            {"state_t": ((1, 2, 3), (1, 2, 3)), "action_t": (1, 2, 3)},
            {"state_t": ((4, 5, 6), (1, 2, 3)), "action_t": (1, 2, 3)},
            {"state_t": ((7, 8, 9), (1, 2, 3)), "action_t": (1, 2, 3)}
        ]
        outputs = [1.0986122886681098, 1.0986122886681098, 1.0986122886681098]

        # Iterate over the input/output pairs and assert that the function
        # produces the expected output for each input
        for (state_action, expected) in zip(inputs, outputs):
            self.assertEqual(calc.calculate_reward(state_action), expected)