from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Предсказание цены")

st.title("Предсказание стоимости жилья в Мумбаи:")

# Пути
MODEL_DIR = "models"
MODEL_NAMES = {
    "Decision Tree Regressor": "best_decision_tree_regressor_gridsearch.joblib",
    "Gradient Boosting": "best_gradient_boosting_regressor.joblib",
    "CatBoost": "catboost_regressor_model.joblib",
    "Bagging": "best_bagging_regressor_gridsearch.joblib",
    "Stacking (DT + ElasticNet)": "best_stacking_regressor_elasticnet_gridsearch.joblib"
}

# Выбор модели
st.sidebar.header("Выберите модель")
model_name = st.sidebar.selectbox("Модель:", list(MODEL_NAMES.keys()))
model_path = os.path.join(MODEL_DIR, MODEL_NAMES[model_name])

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

def make_prediction(df):
    if 'price' in df.columns:
        df = df.drop(columns=['price'])
    prediction = model.predict(df)
    return prediction

# Способ ввода
input_method = st.radio("Как вы хотите ввести данные?", ["Загрузить CSV", "Ввести вручную"])

if input_method == "Загрузить CSV":
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Входные данные:")
        st.dataframe(input_df.head(10))  # Покажем первые 10 строк

        if st.button("Сделать предсказание"):
            preds = make_prediction(input_df)
            preds_rounded = [round(p) for p in preds]
            formatted_preds = [f"{p:,} ₹" for p in preds_rounded]

            # Показ первых 10
            st.write("### Предсказанные цены (первые 10):")
            for i, price in enumerate(formatted_preds[:10], start=1):
                st.write(f"{i}. {price}")

            # Подготовка к скачиванию
            result_df = input_df.copy()
            result_df["Predicted Price (₹)"] = preds_rounded

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать все предсказания (CSV)",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Гистограмма
            st.write("### Распределение предсказанных цен")
            st.bar_chart(pd.Series(preds_rounded).value_counts().sort_index())

else:
    st.write("Введите данные вручную:")
    area = st.number_input("Площадь (кв. футы)", min_value=100, max_value=10000, value=1000)
    latitude = st.number_input("Широта", value=19.0, format="%.6f")
    longitude = st.number_input("Долгота", value=72.8, format="%.6f")
    bedrooms = st.slider("Количество спален", 1, 10, 2)
    bathrooms = st.slider("Количество ванных", 1, 10, 2)
    balcony = st.slider("Балконов", 0, 5, 1)
    status = st.selectbox("Статус (0 - готово, 1 - строится)", [0, 1])
    neworold = st.selectbox("Новостройка? (0 - да, 1 - вторичка)", [0, 1])
    parking = st.slider("Парковочных мест", 0, 5, 1)
    furnished = st.selectbox("Меблировка (0 - нет, 1 - частично, 2 - полная)", [0, 1, 2])
    lift = st.selectbox("Лифт (0 - нет, 1 - есть)", [0, 1])
    building_type = st.selectbox("Тип здания (0 - обычное, 1 - элитное и т.д.)", [0, 1, 2])

    single_input = pd.DataFrame([[
        area, latitude, longitude, bedrooms, bathrooms, balcony,
        status, neworold, parking, furnished, lift, building_type
    ]], columns=[
        'area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms', 'Balcony',
        'Status', 'neworold', 'parking', 'Furnished_status', 'Lift', 'type_of_building'
    ])

    if st.button("Предсказать цену"):
        prediction = make_prediction(single_input)[0]
        st.success(f"Прогнозируемая цена: **{round(prediction):,} ₹**")