import os
import unittest
import pandas as pd
import sys 
import clickhouse_connect

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from preprocess import DataPreprocess


class TestDataPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.data_maker = DataPreprocess()

    def test_get_data(self):
        self.assertEqual(self.data_maker.get_data(), True)

    def test_prepare_data(self):
        self.assertEqual(self.data_maker.prepare_data(), True)

    def test_prepare_target(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='')
        query = client.query("SELECT Text, Category FROM BBC_News_Train")
        dataset = pd.DataFrame(query.result_rows,columns=['Text','Category'])
        client.close()
        
        targets = pd.DataFrame(dataset.Category)
        res = self.data_maker.prepare_labels(targets)
        self.assertEqual(len(self.data_maker.labels_to_id), 5)
        self.assertEqual(len(targets), len(res))

    def test_prepare_text(self):
        client = clickhouse_connect.get_client(host='localhost', username='default', password='')
        query = client.query("SELECT Text, Category FROM BBC_News_Train")
        dataset = pd.DataFrame(query.result_rows,columns=['Text','Category'])
        client.close()
        
        features = pd.DataFrame(dataset.Text)
        res = self.data_maker.prepare_text(features, 'train')
        self.assertEqual(len(features), len(res))
