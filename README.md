# Отчет
Ссылка на github репозиторий: https://github.com/u-tain/BigData_lab_2


1. Сделан форк репозитория на GitHub
2. Clickhouse запущен в докере и интегрирован в код
3. Настроено локальное подключение модели к бд
4. Собран docker-compose.yaml с двумя  сервисами (бд и модель). Сначала удалось установить подключение с бд, но при переходе к следующему шагу (построению ci/cd пайплайну) и последующему возвращению к данному этапу не удалось решить ошибку: 
```
File "/usr/local/lib/python3.10/site-packages/urllib3/util/retry.py", line 515, in incrementapp 
| raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='172.25.0.2', port=8123): 
Max retries exceeded with url: /?wait_end_of_query=1 
(Caused by ConnectTimeoutError(<urllib3.connection.HTTPConnection object at 0x7fcf31a7a710>, 
'Connection to 172.25.0.2 timed out. (connect timeout=10)')
```
5. Был изменен докер образ clickhouse и ошибка ушла
6. Исправлен  CI/CD пайплайн
7. Пароли, имена пользователей и IP адреса оформлены с помощью github secrets

![image](https://github.com/u-tain/BigData_lab_2/assets/43996253/9a5732b7-05af-4d0f-8c7c-868787c01e14)

