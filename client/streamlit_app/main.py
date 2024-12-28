import streamlit as st

from load_dataset import load_dataset
from models import models
from inference import inference
from create_model import create_model

st.set_page_config(page_title="Классификатор текстов", layout="wide")


def main():
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
