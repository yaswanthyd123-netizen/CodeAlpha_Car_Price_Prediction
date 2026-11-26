from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "car_data.csv")
df = df.drop_duplicates()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df = df.dropna(subset=["selling_price", "year"])
df["car_age"] = 2025 - df["year"]
df = df.drop(columns=["year"])

target = "selling_price"
X, y = df.drop(columns=[target]), df[target]
numeric = ["km_driven", "horsepower", "car_age"]
categorical = ["brand", "fuel", "transmission"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "Linear Regression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("prep", preprocessor),
                               ("model", RandomForestRegressor(n_estimators=250, random_state=42))])
}
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    scores[name] = (model, pred, r2)
    print(f"\n{name}\nMAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nR2: {r2:.3f}")

best_name = max(scores, key=lambda n: scores[n][2])
best_model, best_pred, _ = scores[best_name]
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.65)
limits = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual selling price")
plt.ylabel("Predicted selling price")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(out / "actual_vs_predicted.png", dpi=150)
print(f"\nBest model by R2: {best_name}")
print(f"Chart saved in: {out}")
