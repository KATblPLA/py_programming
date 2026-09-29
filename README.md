# ML Project

## Создание окружения
python -m venv .venv
![alt text](image.png)

## Активация скриптов
![alt text](image-1.png)
.venv\Scripts\activate

## Установка uv для более быстрой загрузки пакетов
![alt text](image-2.png)
pip install uv

## Установка зависимостей из заранее заготовленных файлов в папке requirements
uv pip install -r requirements/requirements_ML.txt
uv pip install -r requirements/requirements_service.txt
uv pip install -r requirements/requirements_jupyter.txt

запустится установка
![alt text](image-3.png)

после установки можно проверить
![alt text](image-4.png)

## Запуск приложения FastAPI
запуск:
![alt text](image-5.png)

приложение:
![alt text](image-6.png)

результат:
![alt text](image-7.png)