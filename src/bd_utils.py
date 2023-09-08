import clickhouse_connect
import os

def connect2bd():
    return clickhouse_connect.get_client(host=os.getenv("DB_HOST"), username=os.getenv("DB_USER"), password='',)
