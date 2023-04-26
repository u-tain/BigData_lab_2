import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import configparser


class DataPreprocess():
    def __init__(self, project_path: str = None) -> None:
        if project_path:
            self.project_path = os.path.join(project_path, "data")
        else:
            self.project_path = os.path.join(os.getcwd()[:-4], "data")
        self.config = configparser.ConfigParser()
        self.data_path = os.path.join(self.project_path, "BBC News Train.csv")
        self.test_data_path = os.path.join(self.project_path, "BBC News Test.csv")
        self.config['PROJECT'] = {'path': self.project_path}
        self.config['DATA'] = {'train': self.data_path,
                               'test': self.test_data_path}
        self.labels_to_id = {}
        self.id_to_labels = {}
        self.tfidf = 0
        self.X_path = os.path.join(self.project_path, "features_BBC.csv")
        self.y_path = os.path.join(self.project_path, "targets_BBC.csv")
        self.X_test_path = os.path.join(self.project_path, "features_test_BBC.csv")
        self.train_path = [os.path.join(self.project_path, "Train_features_BBC.csv"), os.path.join(
            self.project_path, "Train_targets_BBC.csv")]
        self.test_path = [os.path.join(self.project_path, "Test_features_BBC.csv")]

    def get_data(self) -> bool:
        dataset = pd.read_csv(self.data_path)
        X = pd.DataFrame(dataset.Text)
        y = pd.DataFrame(dataset.Category)
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        dataset_test = pd.read_csv(self.test_data_path)
        X_test = pd.DataFrame(dataset_test.Text)
        X_test.to_csv(self.X_test_path, index=True)
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path) and os.path.isfile(self.X_test_path):
            return os.path.isfile(self.X_path) and os.path.isfile(self.y_path) and os.path.isfile(self.X_test_path)
        else:
            print("X and y data is not ready")
            return False

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
        self.get_data()
        try:
            X = pd.read_csv(self.X_path, index_col=0)
            y = pd.read_csv(self.y_path, index_col=0)
            X_test = pd.read_csv(self.X_test_path, index_col=0)
        except FileNotFoundError:
            print("data is not found")
            return False
        else:
            print("Данные получены")
        X = self.prepare_text(X, mode='train')
        y = self.prepare_labels(y)
        X_test = self.prepare_text(X_test, mode='test')
        print("Данные готовы")
        self.config['READY_DATA_TRAIN'] = {'X_train': self.train_path[0],
                                           'y_train': self.train_path[1]}
        self.config['READY_DATA_TEST'] = {'X_test': self.test_path[0]}

        self.save_ready_data(X, self.train_path[0], 'Text')
        self.save_ready_data(y, self.train_path[1], 'Category')
        self.save_ready_data(X_test, self.test_path[0], 'Text')

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.train_path[0]) and \
               os.path.isfile(self.train_path[1]) and \
               os.path.isfile(self.test_path[0])

    def save_ready_data(self, arr, path: str, mode: str) -> bool:
        items = arr.tolist()
        df = pd.DataFrame()
        if mode == 'Text':
            df = pd.DataFrame(arr.tolist())
        else:
            df[mode] = items
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        return os.path.isfile(path)


if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    data_preprocess.prepare_data()
