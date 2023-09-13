FROM python:3.10

WORKDIR /app
DB_HOST = ${FOO}
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN ["python", "src/bd_utils.py"]
RUN ["python", "src/preprocess.py"]
RUN ["python", "src/train.py"]
RUN ["python", "src/predict.py"]
RUN ["python", "src/unit_tests/test_preprocess.py"]
RUN ["python", "src/unit_tests/test_train.py"]
RUN ["python", "src/unit_tests/test_predict.py"]
