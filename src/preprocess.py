import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import configparser
import logging
import clickhouse_connect
import numpy as np


class DataPreprocess():
    def __init__(self, project_path: str = None) -> None:
        # подключаемся к базе данных
        self.client = clickhouse_connect.get_client(host='localhost', username='default', password='')
        self.x_table_name = 'Train_features_BBC'
        self.y_table_name = 'targets_BBC'
        self.x_test_table_name = 'Test_features_BBC'

        self.config = configparser.ConfigParser()
        self.config['DATA'] = {'train': 'BBC_News_Train',
                               'test': 'BBC_News_Test'}

    def get_data(self) -> bool:
        # записываем запрос обучающих данных в датафрейм
        query = self.client.query("SELECT Text, Category FROM BBC_News_Train")
        dataset = pd.DataFrame(query.result_rows,columns=['Text','Category'])

        self.X = pd.DataFrame(dataset.Text)
        self.y = pd.DataFrame(dataset.Category)

        # записываем запрос тестовых данных в датафрейм
        query = self.client.query("SELECT * FROM BBC_News_Test")
        dataset = pd.DataFrame(query.result_rows,columns=['idx','Text'])
        self.X_test = pd.DataFrame(dataset.Text)
        

    def prepare_labels(self, targets):
        self.labels_to_id = {key: i for i, key in enumerate(targets.Category.unique())}
        self.id_to_labels = dict(zip(self.labels_to_id.values(), self.labels_to_id.keys()))
        targets.Category = targets.Category.apply(lambda x: self.labels_to_id[x])
        return targets.Category

    def prepare_text(self, features, mode: str):
        if mode == 'train':
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                         stop_words='english')
            features = self.tfidf.fit_transform(features.Text).toarray()
        else:
            features = self.tfidf.transform(features.Text.tolist()).toarray()
        return features

    def prepare_data(self) -> bool:
        try:
            self.get_data()
        except:
            logging.error("Error in get data")
            return False
        else:
            logging.info("Data received")

        X = self.prepare_text(self.X, mode='train')
        y = self.prepare_labels(self.y)
        X_test = self.prepare_text(self.X_test, mode='test')
        logging.info("Data ready")

        self.config['READY_DATA_TRAIN'] = {'X_train': self.x_table_name,
                                           'y_train': self.y_table_name}
        self.config['READY_DATA_TEST'] = {'X_test': self.x_test_table_name}

        # self.save_ready_data(X, self.x_table_name, 'Text')
        # self.save_ready_data(y, self.y_table_name, 'Category')
        self.save_ready_data(X_test, self.x_test_table_name, 'Text')
        logging.info('Data saved')
        
        with open('src/config.ini', 'w') as configfile:
            self.config.write(configfile)
        

    def save_ready_data(self, arr, name: str, mode: str) -> bool:
        items = arr.tolist()
        df = pd.DataFrame()
        if mode == 'Text':
            df = pd.DataFrame(arr.tolist())
        else:
            df[mode] = items
        df = df.reset_index(drop=True)
        # создаем таблицу для результат обработки данных
        print(df.columns)
        if 'Range' in str(df.columns):
            columns = np.arange(df.columns.start,df.columns.stop)
        else: 
            columns = df.columns
        num_columns = len(columns)
        columns = [f'"{item}" FLOAT' for item in columns]
        columns = str(columns).replace('[','').replace(']','').replace("'","")
        text_query = f'CREATE TABLE  IF NOT EXISTS {name}  ({columns}) ENGINE = Log'
        delete_query = f'DROP TABLE {name};'
        self.client.query(text_query)
        rows = df.values.tolist() 
        # print(rows[0])
        rows = str(rows)[1:-1].replace('[','(').replace(']',')').replace('\n','')
        # print(name)
        self.client.query(f'SET memory_overcommit_ratio_denominator=4000, memory_usage_overcommit_max_wait_microseconds=500')
        insert_query = f'INSERT INTO {name}  VALUES {rows} '
        print(self.client.query(insert_query))
        print(self.client.query(f'SELECT * FROM {name}').result_rows)
        # self.client.query(insert_query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="preprocess.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    data_preprocess = DataPreprocess()
    data_preprocess.prepare_data()
