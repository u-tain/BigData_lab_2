import clickhouse_connect
import os

def connect2bd():
    return clickhouse_connect.get_client(host='172.17.0.2', username=os.getenv("DB_USER"), password='',)
