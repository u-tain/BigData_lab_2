import pandas as pd
import pickle
import os
import configparser


class Predictor():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
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
            print("Model wasn't trained")
            return False
        Y_pred = classifier.predict(self.X_test)
        Y_pred = self.post_process(Y_pred)
        results = pd.DataFrame({
            "ArticleId": self.test_df_before_prepoc["ArticleId"],
            "Category": Y_pred
        })
        print('Предсказания выполнены')
        self.config['RESULT'] = {'path': self.result_path}
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        results.to_csv(self.result_path, index=False)
        return True

    def post_process(self, predictions):
        pred_names = []
        for item in predictions:
            pred_names.append(self.id_to_labels[item])
        return pred_names


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
