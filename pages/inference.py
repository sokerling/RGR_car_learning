import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Предсказание цены")

st.title("Предсказание стоимости жилья в Мумбаи")

MODEL_DIR = "models"
MODEL_NAMES = {
    "Decision Tree Regressor": "best_decision_tree_regressor_gridsearch.joblib",
    "Gradient Boosting": "best_gradient_boosting_regressor_gridsearch.joblib",
    "CatBoost": "catboost_regressor_model.joblib",
    "Bagging": "best_bagging_regressor_gridsearch.joblib",
    "Stacking (DT + ElasticNet)": "best_stacking_regressor_elasticnet_gridsearch.joblib",
    "Neural Network": "best_nuero_rand.joblib"
}

st.sidebar.header("Выберите модели")
selected_models = st.sidebar.multiselect("Модели:", list(MODEL_NAMES.keys()), default=list(MODEL_NAMES.keys())[:1])

def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели {path}: {e}")
        return None

# Выбор способа ввода данных
input_method = st.radio("Как вы хотите ввести данные?", ["Загрузить CSV", "Ввести вручную"])

if input_method == "Загрузить CSV":
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Входные данные:")
        st.dataframe(input_df.head(10))

        if st.button("Сделать предсказание"):
            if not selected_models:
                st.warning("Пожалуйста, выберите хотя бы одну модель.")
            else:
                try:
                    preds_df = pd.DataFrame(index=input_df.index)

                    for model_name in selected_models:
                        model_path = os.path.join(MODEL_DIR, MODEL_NAMES[model_name])
                        model = load_model(model_path)
                        if model is None:
                            st.error(f"Модель {model_name} не загружена, пропускаем.")
                            continue
                        df_for_pred = input_df.copy()
                        if 'price' in df_for_pred.columns:
                            df_for_pred = df_for_pred.drop(columns=['price'])
                        preds = model.predict(df_for_pred)
                        preds_rounded = [round(p) for p in preds]
                        preds_df[model_name] = preds_rounded

                    st.write("### Предсказанные цены (первые 10 строк):")
                    st.dataframe(preds_df.head(10))

                    # Вывод распределения предсказаний по каждой модели
                    st.write("### Распределение предсказанных цен по моделям:")
                    for model_name in preds_df.columns:
                        st.write(f"**{model_name}**")
                        st.bar_chart(pd.Series(preds_df[model_name]).value_counts().sort_index())

                    result_df = pd.concat([input_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать все предсказания (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Ошибка при предсказании: {e}")

else:
    st.write("Введите данные вручную:")
    area = st.number_input("Площадь (кв. футы)", min_value=100, max_value=10000, value=1000)
    latitude = st.number_input("Широта", value=19.0, format="%.6f")
    longitude = st.number_input("Долгота", value=72.8, format="%.6f")
    bedrooms = st.slider("Количество спален", 1, 10, 2)
    bathrooms = st.slider("Количество ванных", 1, 10, 2)
    balcony = st.slider("Балконов", 0, 5, 1)

    status_map = {"Готово": 0, "Строится": 1}
    status = st.selectbox("Статус объекта", list(status_map.keys()))

    neworold_map = {"Новостройка": 0, "Вторичка": 1}
    neworold = st.selectbox("Тип жилья", list(neworold_map.keys()))

    parking = st.slider("Парковочных мест", 0, 5, 1)

    furnished_map = {"Нет": 0, "Частично": 1, "Полная": 2}
    furnished = st.selectbox("Меблировка", list(furnished_map.keys()))

    lift_map = {"Без лифта": 0, "С лифтом": 1}
    lift = st.selectbox("Лифт", list(lift_map.keys()))

    building_type_map = {"Обычное": 0, "Элитное": 1}
    building_type = st.selectbox("Тип здания", list(building_type_map.keys()))

    # Собираем числовые коды из выбранных текстов
    single_input = pd.DataFrame([[
        area,
        latitude,
        longitude,
        bedrooms,
        bathrooms,
        balcony,
        status_map[status],
        neworold_map[neworold],
        parking,
        furnished_map[furnished],
        lift_map[lift],
        building_type_map[building_type]
    ]], columns=[
        'area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms', 'Balcony',
        'Status', 'neworold', 'parking', 'Furnished_status', 'Lift', 'type_of_building'
    ])

    if st.button("Предсказать цену"):
        if not selected_models:
            st.warning("Пожалуйста, выберите хотя бы одну модель.")
        else:
            try:
                preds = {}
                for model_name in selected_models:
                    model_path = os.path.join(MODEL_DIR, MODEL_NAMES[model_name])
                    model = load_model(model_path)
                    if model is None:
                        st.error(f"Модель {model_name} не загружена, пропускаем.")
                        continue
                    pred = model.predict(single_input)[0]
                    preds[model_name] = round(pred)

                st.write("### Предсказанные цены:")
                st.table(pd.DataFrame([preds]))

            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
