import os
import unittest
import pandas as pd
from src.preprocess import DataPreprocess


class TestDataPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.data_maker = DataPreprocess(os.getcwd()[:-14])

    def test_get_data(self):
        self.assertEqual(self.data_maker.get_data(), True)

    def test_prepare_data(self):
        self.assertEqual(self.data_maker.prepare_data(), True)

    def test_prepare_target(self):
        project_path = os.path.join(os.getcwd()[:-14], "data")
        data_path = os.path.join(project_path, "BBC News Train.csv")
        targets = pd.read_csv(data_path)
        res = self.data_maker.prepare_labels(targets)
        self.assertEqual(len(self.data_maker.labels_to_id), 5)
        self.assertEqual(len(targets), len(res))

    def test_prepare_text(self):
        project_path = os.path.join(os.getcwd()[:-14], "data")
        data_path = os.path.join(project_path, "BBC News Train.csv")
        features = pd.read_csv(data_path)
        res = self.data_maker.prepare_text(features, 'train')
        self.assertEqual(len(features), len(res))
