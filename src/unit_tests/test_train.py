import unittest
from src.train import Model


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = Model()

    def test_log_reg(self):
        self.assertEqual(self.trainer.log_reg(), True)

