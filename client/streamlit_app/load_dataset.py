import streamlit as st
import pandas as pd
import plotly.express as px
import io


def load_dataset():
    st.header("Загрузка данных и EDA")

    df = st.session_state.df

    uploaded_file = st.file_uploader("Загрузить CSV файл", type=['csv'])
    st.session_state['uploaded_file'] = uploaded_file

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

    if df is not None:
        display_data(df)
        perform_eda(df)


def display_data(df):
    st.write("### Просмотр данных")
    st.dataframe(df.head())


def perform_eda(df):
    st.write("### Разведочный анализ данных (EDA)")

    st.write("#### Статистическое описание")
    st.write(df.describe())

    st.write("#### Визуализации")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f'Распределение {col}')
        st.plotly_chart(fig)

    if len(numeric_cols) > 1:
        st.write("#### Корреляционная матрица")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title='Корреляционная матрица')
        st.plotly_chart(fig)
