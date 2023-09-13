import clickhouse_connect
import os,sys
import pandas as pd

HOST = 'localhost' # os.getenv("DB_HOST")
USER = 'default' #os.getenv("DB_USER")
def connect2bd():
    return clickhouse_connect.get_client(host=os.getenv("DB_HOST"), username=os.getenv("DB_USER"), password='', )

def upload_data():
    client = connect2bd()
    # загружаем в базу данных данные для обучения
    querry1 = 'CREATE TABLE IF NOT EXISTS BBC_News_Train ( `Articled` Int, `Text` String, `Category` String) ENGINE = MergeTree ORDER BY Articled'
    client.query(querry1)
    df = pd.read_csv(os.path.join(os.getcwd(),'data','BBC News Train.csv'))
    rows = list(df.itertuples(index=False, name=None))
    insert_querry1 = 'INSERT INTO BBC_News_Train VALUES '+str(rows).replace('[','').replace(']','')
    client.query(insert_querry1)

    # загружаем в базу данных данные для теста
    
    querry2 = 'CREATE TABLE IF NOT EXISTS BBC_News_Test ( `Articled` Int, `Text` String) ENGINE = MergeTree ORDER BY Articled'
    client.query(querry2)
    df = pd.read_csv(os.path.join(os.getcwd(),'data','BBC News Test.csv'))
    rows = list(df.itertuples(index=False, name=None))
    insert_querry2 = 'INSERT INTO BBC_News_Test VALUES '+str(rows).replace('[','').replace(']','')
    client.query(insert_querry2)


if __name__ == "__main__":
    upload_data()
