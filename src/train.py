import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import configparser
import logging
import clickhouse_connect



class Model():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("src/config.ini")
        self.prodject_path = self.project_path = os.getcwd().replace('\\','/')

        # подключаемся к базе данных
        self.client = clickhouse_connect.get_client(host='localhost', username='default', password='',)

        query1= self.client.query(f"SELECT * FROM {self.config['READY_DATA_TRAIN']['x_train']}")
        query2= self.client.query(f"SELECT * FROM {self.config['READY_DATA_TRAIN']['y_train']}")
        df1  = pd.DataFrame(columns= np.arange(int(self.config['READY_DATA_TRAIN']['x_train_columns'])),)
        df2 = pd.DataFrame(columns = ['Category'])
        rows1 = query1.result_rows
        rows2 = query2.result_rows
        for i in range(len(rows1)):
            df1.loc[len(df1)] = rows1[i]
            df2.loc[len(df2)] = rows2[i]

        self.X_train = [df1.iloc[i, :].array for i in range(len(df1))]
        self.y_train = df2.Category

        self.log_reg_path = os.path.join( self.prodject_path,'experiments', "logreg.sav")
        self.client.close()


    def log_reg(self) -> bool:
        classifier = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=0)
        logging.info('the model has been initialized')
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            logging.error("Something went wrong in fit model")
        else:
            logging.info('Model successfully trained')
        params = classifier.get_params()
        self.config['LOGREG'] = {'penalty': params['penalty'],
                                 'C': params['C'],
                                 'max_iter': params['max_iter'],
                                 'random_state': params['random_state'],
                                 'model_path': self.log_reg_path
                                 }
        with open('src/config.ini', 'w') as configfile:
            self.config.write(configfile)
        return self.save_model(classifier, self.log_reg_path)

    def save_model(self, classifier, path: str) -> bool:
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        logging.info('model saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="train.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    multi_model = Model()
    multi_model.log_reg()
