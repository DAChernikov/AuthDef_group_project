import io
import json

import requests
import logging
import streamlit as st

from constants import API_URL

logger = logging.getLogger(__name__)


def create_model():
    """
    Интерфейс Streamlit для обучения модели.

    Проверяет, загружены ли данные, и предоставляет форму для ввода параметров модели.
    Если данные не загружены, выводится предупреждение.
    В случае успешного обучения модели отображается сообщение с результатами.
    """
    st.header("Обучение модели")

    if st.session_state.df is None:
        st.warning("Сначала загрузите данные")
    else:
        with st.form("model_training_form"):
            model_id = st.text_input("Имя модели")
            model_type = st.selectbox("Тип модели", ["logistic"])

            st.subheader("Гиперпараметры")
            solver = st.selectbox("Solver", ["liblinear"])
            max_iter = st.slider("Maximum iterations", 100, 1000, 100)

            submit_button = st.form_submit_button("Обучить модель")

            if submit_button:
                if not model_id:
                    st.error("Пожалуйста, введите имя модели")
                else:
                    with st.spinner("Обучение модели..."):
                        result = train_model(
                            st.session_state.df,
                            model_id,
                            model_type,
                            solver,
                            max_iter
                        )
                        if result:
                            st.success("Модель успешно обучена!")
                            st.json(result)


def train_model(df, model_id, model_type, solver, max_iter):
    """
    Обучает модель на основе переданных данных и гиперпараметров.

    Отправляет данные и параметры на сервер для обучения модели и возвращает результат.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()

    files = {"file": csv_str}
    data = {
        "target_column": "target",
        "config": json.dumps({
            "id": model_id,
            "ml_model_type": model_type,
            "hyperparameters": {
                "solver": solver,
                "max_iter": max_iter
            }
        })
    }

    try:
        response = requests.post(f"{API_URL}/api/v1/models/fit_csv", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(e)
        st.error("Ошибка при обучении модели")
        return
