import pandas as pd
import pickle
import os

class Predictor():
    def __init__(self, project_path: str = None) -> None:
        if project_path:
            self.project_path = os.path.join(project_path, "data")
        else:
            self.project_path = os.path.join(os.getcwd()[:-4], "data")
        df = pd.read_csv(os.path.join(self.project_path, 'Test_features_BBC.csv'), index_col=0)
        self.X_test = [df.iloc[i, :].array for i in range(len(df))]
        self.model_path = os.path.join(self.project_path[:-5], 'experiments/logreg.sav')
        self.test_df_before_prepoc = pd.read_csv(os.path.join(self.project_path, 'BBC News Test.csv'))
        self.Train = pd.read_csv(os.path.join(self.project_path, "BBC News Train.csv"), index_col=0)
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
