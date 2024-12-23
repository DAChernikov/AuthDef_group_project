import json
import os

HISTORY_FILE = "model_history.json"  # Имя и путь файла-регистратора моделей

def load_model_history():
    """Загружает историю моделей из файла-регистратора."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_model_history(history):
    """Сохраняет историю моделей в файл-регистратор моделей."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)