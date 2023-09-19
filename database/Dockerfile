FROM clickhouse/clickhouse-server

ENV ulimit nofile=262144:262144
ENV CLICKHOUSE_DB default 
ENV CLICKHOUSE_USER  utain 
ENV CLICKHOUSE_PASSWORD 3244 
EXPOSE 8123:8123

COPY ./database/config.xml /etc/clickhouse-server/config.xml
 



