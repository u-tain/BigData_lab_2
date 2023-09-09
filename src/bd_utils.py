import clickhouse_connect
import os
HOST = '172.17.0.2' # os.getenv("DB_HOST")
USER = os.getenv("DB_USER")
print(HOST,USER)
def connect2bd():
    return clickhouse_connect.get_client(host=HOST, username=USER, password='',)
