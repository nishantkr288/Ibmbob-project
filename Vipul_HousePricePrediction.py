"""
House Price Prediction using Linear Regression
Author: Vipul Shukla
Project: House Price Prediction
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_price.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
SCALER_PATH = BASE_DIR / "house_price_scaler.pkl"

def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    required = {"area", "rooms", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    print("Dataset shape:", df.shape)
    print("\nSummary statistics:\n", df.describe())

    # 2. Features and target
    X = df[["area", "rooms"]].values
    y = df["price"].values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 4. Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_sc)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    # 7. Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("\nSaved:", MODEL_PATH.name, "and", SCALER_PATH.name)

    # 8. Example prediction
    example = pd.DataFrame({"area": [2000], "rooms": [3]})
    example_sc = scaler.transform(example)
    estimated_price = model.predict(example_sc)[0]
    print(f"\nExample: 2000 sq ft, 3 rooms -> estimated price = {estimated_price:,.0f}")

if __name__ == "__main__":
    main()
    