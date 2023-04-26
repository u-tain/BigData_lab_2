import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import configparser


class Model():
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.project_path = self.config['PROJECT']['path']

        df = pd.read_csv(self.config['READY_DATA_TRAIN']['X_train'], index_col=0)
        self.X_train = [df.iloc[i, :].array for i in range(len(df))]
        self.y_train = pd.read_csv(self.config['READY_DATA_TRAIN']['y_train'], index_col=0).Category
        self.project_path = os.path.join(self.project_path[:-5], "experiments")
        self.log_reg_path = os.path.join(self.project_path, "logreg.sav")

    def log_reg(self) -> bool:
        classifier = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=0)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            print("Something went wrong")
        else:
            print('Модель успешно обучена')
        params = classifier.get_params()
        self.config['LOGREG'] = {'penalty': params['penalty'],
                                 'C': params['C'],
                                 'max_iter': params['max_iter'],
                                 'random_state': params['random_state'],
                                 'model_path': self.log_reg_path
                                 }
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return self.save_model(classifier, self.log_reg_path)

    def save_model(self, classifier, path: str) -> bool:
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = Model()
    multi_model.log_reg()
