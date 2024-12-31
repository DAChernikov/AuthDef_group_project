import streamlit as st

from load_dataset import load_dataset
from models import models
from inference import inference
from create_model import create_model

st.set_page_config(page_title="Классификатор текстов", layout="wide")


def main():
    """
    Главная функция приложения Streamlit.

    Инициализирует сессию с данными и предоставляет пользователю выбор различных разделов
    через боковую панель. В зависимости от выбранного раздела, отображается соответствующая
    страница для работы с моделями, загрузки данных, создания модели или выполнения инференса.

    Разделы:
    - Модели: Страница для работы с моделями.
    - Загрузка данных и EDA: Страница для загрузки данных и выполнения разведочного анализа.
    - Создание модели: Страница для создания новой модели.
    - Инференс: Страница для выполнения инференса на основе обученной модели.
    """
    if 'df' not in st.session_state:
        st.session_state.df = None

    pages = {
        "Модели": models,
        "Загрузка данных и EDA": load_dataset,
        "Создание модели": create_model,
        "Инференс": inference,
    }

    st.sidebar.title("Навигация")
    selection = st.sidebar.radio(label="Выберите раздел", options=list(pages.keys()))
    pages[selection]()


if __name__ == "__main__":
    main()
