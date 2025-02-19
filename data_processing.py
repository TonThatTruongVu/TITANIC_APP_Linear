import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import joblib
import mlflow
import mlflow.sklearn

def load_and_process_data(url, test_size=0.15, val_size=0.15, poly_degree=1):
    # Đọc dữ liệu
    df = pd.read_csv(url)
    
    # Chọn các cột cần thiết
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # Chọn đặc trưng
    
    # Xử lý dữ liệu bị thiếu
    df.dropna(inplace=True)
    
    # Chuyển đổi categorical thành numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Chia dữ liệu thành đầu vào X và đầu ra y
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Chia dữ liệu thành tập train, validation, test
    train_size = 1 - (test_size + val_size)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Lưu scaler để sử dụng trong app.py
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Biến đổi Polynomial Features nếu cần
    if poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)
        X_test = poly.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def train_model(X_train, y_train, poly_degree=1):
    mlflow.set_experiment("Titanic Regression Experiment")
    with mlflow.start_run():
        model = LinearRegression()
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        model.fit(X_train, y_train)
        
        mlflow.log_param("poly_degree", poly_degree)
        mlflow.log_metric("mean_r2", scores.mean())
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        #Thêm input_example để MLflow nhận diện dữ liệu đầu vào
        input_example = np.array([X_train[0]])  # Lấy một dòng dữ liệu đầu vào

        mlflow.sklearn.log_model(model, "model", input_example=input_example)  # Cập nhật đoạn này
        
    return model

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_process_data(url)
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # Huấn luyện mô hình
    train_model(X_train, y_train)
