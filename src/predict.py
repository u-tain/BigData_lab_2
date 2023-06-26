import pandas as pd
import pickle
import os
import configparser
import logging

class Predictor():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("src/config.ini")
        self.project_path = self.config['PROJECT']['path']

        df = pd.read_csv(self.config['READY_DATA_TEST']['x_test'], index_col=0)
        self.X_test = [df.iloc[i, :].array for i in range(len(df))]
        self.model_path = self.config['LOGREG']['model_path']
        self.test_df_before_prepoc = pd.read_csv(self.config['DATA']['test'])
        self.Train = pd.read_csv(self.config['DATA']['train'], index_col=0)
        self.labels_to_id = {key: i for i, key in enumerate(self.Train.Category.unique())}
        self.id_to_labels = dict(zip(self.labels_to_id.values(), self.labels_to_id.keys()))
        self.result_path = os.path.join(self.project_path[:-5], 'experiments/result.csv')

    def predict(self) -> bool:
        try:
            classifier = pickle.load(open(self.model_path, "rb"))
        except FileNotFoundError:
            logging.error("Model wasn't trained")
            return False
        logging.info('Start predictions')
        Y_pred = classifier.predict(self.X_test)
        logging.info('Predictions fulfilled')
        Y_pred = self.post_process(Y_pred)
        logging.info('Predictions post-processed')
        results = pd.DataFrame({
            "ArticleId": self.test_df_before_prepoc["ArticleId"],
            "Category": Y_pred
        })
        self.config['RESULT'] = {'path': self.result_path}
        with open('src/config.ini', 'w') as configfile:
            self.config.write(configfile)
        results.to_csv(self.result_path, index=False)
        logging.info('results written to file ' + self.result_path)
        return True

    def post_process(self, predictions):
        pred_names = []
        for item in predictions:
            pred_names.append(self.id_to_labels[item])
        return pred_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="predict.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    predictor = Predictor()
    predictor.predict()
