# Отчет
Ссылка на github репозиторий: https://github.com/u-tain/BigData_lab_2

Ссылка на docker image: https://hub.docker.com/r/utainsacc/clockbox

1. Создан репозиторий на GitHub
2. Данные, доставшиеся согласно варианту задания - текста, набор данных отлично подходит для решения задачи классификации. Произведена предобработка данных.
3. Разработана ML модель с Логистической регрессией
4. Конвертирована модель из *.ipynb в .py скрипты. 
   Они находятся в папке src: preprocess.py, train.py, predict.py
6. С помощью библиотеки unittest код был покрыт тестами, расположение: src/unit_tests
7. Задействован DVC 
8. Создан докер образ
9. создан Dockerfile с указанием версии питона, установкой необходимых библиотек, запуска пайплайнов и тестов внутри докера. Создан и наполнен docker-compose.yaml
10. создан CI pipeline, его код: 
   ![image](https://github.com/u-tain/BigData_lab_1/assets/43996253/c3306b11-6f27-4de9-a17f-1829e92d7813)

11. создан CD pipeline, его код: 

    ![image](https://github.com/u-tain/BigData_lab_1/assets/43996253/e288aa5f-316f-42d3-955a-5f53a3e94851)

Результат работы пайплайна и тестирования:
![image](https://github.com/u-tain/BigData_lab_1/assets/43996253/1e304c9f-a59d-4f9d-af88-d6545732a7b7)
