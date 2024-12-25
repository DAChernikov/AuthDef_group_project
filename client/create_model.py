import io
import json

import requests
import streamlit as st

from constants import API_URL


def create_model():
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
                "solver": "liblinear",
                "max_iter": max_iter
            }
        })
    }

    try:
        response = requests.post(f"{API_URL}/api/v1/models/fit_csv", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при обучении модели: {str(e)}")
        return None
