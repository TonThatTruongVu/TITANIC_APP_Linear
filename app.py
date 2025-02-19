import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd
import joblib

# Load mô hình từ MLflow
logged_model = "runs:/36ae82a8bfa542cf8c1bfdff2583a93f/model"
model = mlflow.pyfunc.load_model(logged_model)

# Load scaler để chuẩn hóa dữ liệu
scaler = joblib.load("models/scaler.pkl")  # Đảm bảo file này tồn tại

# Streamlit UI
st.title("Dự đoán khả năng sống sót Titanic")

# Nhập thông tin hành khách
pclass = st.selectbox("Hạng vé (1: First, 2: Second, 3: Third)", [1, 2, 3])
sex = st.radio("Giới tính", ["Nam", "Nữ"])
age = st.number_input("Tuổi", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Số anh chị em / vợ chồng đi cùng", min_value=0, max_value=10, value=0)
parch = st.number_input("Số cha mẹ / con cái đi cùng", min_value=0, max_value=10, value=0)
fare = st.number_input("Giá vé", min_value=0.0, max_value=500.0, value=50.0)

# Chuyển đổi dữ liệu đầu vào
sex = 1 if sex == "Nữ" else 0
input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], 
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

# Chuẩn hóa dữ liệu đầu vào
input_data_scaled = scaler.transform(input_data)

# Dự đoán
if st.button("Dự đoán"):
    prediction = model.predict(pd.DataFrame(input_data_scaled, 
                                            columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]))[0]
    st.write(f"### Xác suất sống sót dự đoán: {round(prediction, 2)}")
