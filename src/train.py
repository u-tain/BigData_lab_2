import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


class Model():
    def __init__(self, project_path: str = None) -> None:
        if project_path:
            self.project_path = os.path.join(project_path, "data")
        else:
            self.project_path = os.path.join(os.getcwd()[:-4], "data")
        df = pd.read_csv(os.path.join(self.project_path, 'Train_features_BBC.csv'), index_col=0)
        self.X_train = [df.iloc[i, :].array for i in range(len(df))]
        self.y_train = pd.read_csv(os.path.join(self.project_path, 'Train_targets_BBC.csv'), index_col=0).Category
        self.project_path = os.path.join(self.project_path[:-5], "experiments")
        self.log_reg_path = os.path.join(self.project_path, "logreg.sav")

    def log_reg(self) -> bool:
        classifier = LogisticRegression(random_state=0)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            print("Something went wrong")
        return self.save_model(classifier, self.log_reg_path)

    def save_model(self, classifier, path: str) -> bool:
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = Model()
    multi_model.log_reg()
