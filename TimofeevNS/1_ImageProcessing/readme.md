tbd

Инструкции по работе с подпапкой:
всё - через единый CLI — `run-filters.py`.

## Содержание

- [О проекте](#о-проекте)
- [Что понадобится](#что-понадобится)
- [Установка](#установка)
  - [Linux](#linux-ubuntu--debian--mint)
  - [macOS](#macos)
  - [Windows](#windows)
- [Проверка установки](#проверка-установки)
- [Использование `run-filters.py`](#использование-run-filterspy)

## О проекте

Структура:

- `image_filters.py` — библиотека фильтров (шум, сепия, царапины, виньетки, эффекты плёнки, бинаризация Оцу, выделение краёв и др.)
- `run-filters.py` — CLI для применения фильтров к изображению
- `imgs/` — примеры входных изображений

Изображения обрабатываются в BGR (нотация OpenCV), тип `uint8`, форма `(H, W, 3)`.

## Что понадобится

- Python **3.9 или новее** (рекомендуется 3.10+)
- pip (обычно идёт в комплекте с Python)
- git (для клонирования репозитория)
- ~50 МБ свободного места под зависимости

Зависимости Python:
- `numpy` — матричные операции
- `opencv-python` — чтение/запись изображений

## Установка

### Linux (Ubuntu / Debian / Mint)

#### Шаг 1. Python и системные зависимости

sudo apt update
sudo apt install python3 python3-pip python3-venv git libgl1 libglib2.0-0

    python3-venv нужен для модуля venv

    libgl1 и libglib2.0-0 — обязательны для opencv-python, иначе import cv2 падает с libGL.so.1: cannot open shared object file

Шаг 2. Клонирование

git clone <URL_репозитория>
cd <папка_репозитория>

Если репозиторий приватный и вы работаете через SSH — настройте ключи заранее. Для HTTPS git попросит логин/токен.
Шаг 3. Виртуальное окружение

python3 -m venv venv
source venv/bin/activate

После source в приглашении появится (venv) — значит, python, pip теперь указывают на изолированную версию из venv/.

При следующем открытии терминала активацию нужно повторить. Деактивация — deactivate.
Шаг 4. Зависимости

pip install --upgrade pip
pip install numpy opencv-python

macOS
Шаг 1. Python и git

На macOS обычно уже есть python3, но не всегда актуальной версии. Рекомендуется поставить через Homebrew:

# Если Homebrew нет:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python и git:
brew install python git

Проверка: python3 --version (должен быть 3.9+).
Шаг 2. Клонирование

git clone <URL_репозитория>
cd <папка_репозитория>

Шаг 3. Виртуальное окружение

python3 -m venv venv
source venv/bin/activate

Синтаксис активации совпадает с Linux. На Apple Silicon (M1/M2/M3) специальных действий не нужно — numpy и opencv-python имеют готовые wheels под arm64.
Шаг 4. Зависимости

pip install --upgrade pip
pip install numpy opencv-python

Windows
Шаг 1. Python и git

Скачайте установщик с https://www.python.org/downloads/ и запустите.

⚠️ Важно: на первом экране установщика обязательно отметьте галочку «Add Python to PATH» — иначе python из терминала не найдётся.

git — с https://git-scm.com/download/win.

Проверка:

python --version
git --version

Если python не находится, попробуйте py --version (лаунчер Windows). В дальнейшем вместо python можно использовать py.
Шаг 2. Клонирование

В PowerShell или CMD:

git clone <URL_репозитория>
cd <папка_репозитория>

Шаг 3. Виртуальное окружение

python -m venv venv

Активация зависит от терминала:

.\venv\Scripts\Activate.ps1

Если PowerShell блокирует запуск скриптов с ошибкой running scripts is disabled, разрешите локальные скрипты (один раз):

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

CMD:
cmd

venv\Scripts\activate.bat

Git Bash / MSYS2:

source venv/Scripts/activate

После активации приглашение станет вида (venv) C:\...>. Деактивация — deactivate.
Шаг 4. Зависимости

pip install --upgrade pip
pip install numpy opencv-python

Проверка установки

Из активированного venv:

python -c "import cv2, numpy; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__)"

Должно вывести две строки с версиями без ошибок.
Использование run-filters.py

Скрипт применяет один или несколько фильтров к изображению и сохраняет результаты в отдельные файлы с именами <имя_входа>_<имя_фильтра>.png.
Базовый запуск

python image_filters_project/run-filters.py <путь к фото>

Без флага -f применяются все зарегистрированные фильтры (~22 штуки). Результаты сохраняются в ./output/.

Только отдельные фильтры:

python image_filters_project/run-filters.py <путь к фото> -f sepia -f aging

