import os

# Получаем токен бота из переменной окружения BOT_TOKEN
BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Необходимо установить переменную окружения BOT_TOKEN!")

# Получаем URL базы данных (если используем базу данных) из переменной окружения DATABASE_URL
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///audio_bot.db")  # Значение по умолчанию - SQLite

# Путь к папке для сохранения аудиофайлов (можно задать через переменную окружения)
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "audio_uploads")