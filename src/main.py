from bd_utils import *
from preprocess import *
from train import *
from predict import *
from unit_tests.test_preprocess import *
from unit_tests.test_train import *
from unit_tests.test_predict import *



if __name__ == "__main__":
    upload_data()
    print('loaded')
    DataPreprocess().prepare_data()
    print('preprocessed')
    Model().log_reg()
    print('trained')
    Predictor().predict()
    print('predicted')
    # запускаем тесты
    TestDataPreprocess()()
    TestTrain()()
    TestPredictor()()
