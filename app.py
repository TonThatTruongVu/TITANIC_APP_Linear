import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load mô hình từ file thay vì MLflow
model_path = "models/model.pkl"
scaler_path = "models/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Không tìm thấy model hoặc scaler. Hãy kiểm tra lại!")

# Streamlit UI
st.title("🚢 Dự đoán khả năng sống sót Titanic")

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

# Dự đoán khi bấm nút
if st.button("🔍 Dự đoán"):
    try:
        # Chuẩn hóa dữ liệu đầu vào
        input_data_scaled = scaler.transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_data_scaled)[0]
