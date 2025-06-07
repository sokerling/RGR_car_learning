import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(page_title="Предсказание цены")

st.title("Предсказание стоимости жилья в Мумбаи")

# Папка с моделями и их имена
MODEL_DIR = "models"
MODEL_NAMES = {
    "Decision Tree Regressor": "best_decision_tree_regressor_gridsearch.joblib",
    "Gradient Boosting": "best_gradient_boosting_regressor_gridsearch.joblib",
    "CatBoost": "catboost_regressor_model.joblib",
    "Bagging": "best_bagging_regressor_gridsearch.joblib",
    "Stacking (DT + ElasticNet)": "best_stacking_regressor_elasticnet_gridsearch.joblib"
}

st.sidebar.header("Выберите модели")
selected_models = st.sidebar.multiselect(
    "Модели для предсказания:",
    options=list(MODEL_NAMES.keys()),
    default=["Gradient Boosting"]
)

def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели `{path}`: {e}")
        return None

def make_prediction(model, df):
    if 'price' in df.columns:
        df = df.drop(columns=['price'])
    prediction = model.predict(df)
    return prediction


# Способ ввода данных
input_method = st.radio("Как вы хотите ввести данные?", ["Загрузить CSV", "Ввести вручную"])

if st.button("Сделать предсказание"):
    if not selected_models:
        st.warning("Пожалуйста, выберите хотя бы одну модель.")
    else:
        predictions = {}
        for model_name in selected_models:
            model_path = os.path.join(MODEL_DIR, MODEL_NAMES[model_name])
            model = load_model(model_path)
            if model:
                try:
                    preds = make_prediction(model, input_df)
                    predictions[model_name] = [round(p) for p in preds]
                except Exception as e:
                    st.error(f"Ошибка при предсказании ({model_name}): {e}")

        if predictions:
            # Объединяем входные данные и предсказания
            results_df = input_df.copy()
            for model_name, preds in predictions.items():
                results_df[model_name] = preds

            st.write("### Результаты предсказаний (первые 10 строк):")
            st.dataframe(results_df.head(10))

            # Кнопка загрузки
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать предсказания (CSV)",
                data=csv,
                file_name="multi_model_predictions.csv",
                mime="text/csv"
            )

            # Гистограммы по моделям
            st.write("### Гистограмма предсказаний по каждой модели:")
            for model_name in selected_models:
                st.write(f"Модель: {model_name}")
                st.bar_chart(pd.Series(predictions[model_name]).value_counts().sort_index())

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

    single_input = pd.DataFrame([[area, latitude, longitude, bedrooms, bathrooms, balcony,
                                  status, neworold, parking, furnished, lift, building_type]],
                                columns=['area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms', 'Balcony',
                                         'Status', 'neworold', 'parking', 'Furnished_status', 'Lift', 'type_of_building'])

    if st.button("Предсказать цену"):
        if not selected_models:
            st.warning("Пожалуйста, выберите хотя бы одну модель.")
        else:
            for model_name in selected_models:
                model_path = os.path.join(MODEL_DIR, MODEL_NAMES[model_name])
                model = load_model(model_path)
                if model:
                    try:
                        prediction = make_prediction(model, single_input)[0]
                        st.success(f"{model_name}: **{round(prediction):,} ₹**")
                    except Exception as e:
                        st.error(f"Ошибка при предсказании ({model_name}): {e}")