import os

os.makedirs("static/images", exist_ok=True)
import pandas as pd
import numpy as np
# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("traffic_demand_dataset1.csv")

print("Dataset Shape:", df.shape)
print(df.head())
# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# Traffic Demand Over Time
# -----------------------------

import matplotlib.pyplot as plt

# Average traffic demand per day
daily_traffic = df.groupby(df["timestamp"].dt.date)["traffic_demand"].mean()

plt.figure(figsize=(15,6))
plt.plot(daily_traffic.index, daily_traffic.values)

plt.title("Average Traffic Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Average Traffic Demand")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("static/images/traffic_demand_over_time.png")
plt.show()
# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create additional time features
df["year"] = df["timestamp"].dt.year
df["day"] = df["timestamp"].dt.day

# Drop timestamp
df.drop("timestamp", axis=1, inplace=True)
df["event_type"] = df["event_type"].fillna("None")

# Encode categorical columns
# Encode categorical columns

encoders = {}

categorical_columns = [
    "road_type",
    "day_of_week",
    "weather_condition",
    "event_type",
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Target

selected_features = [
    "road_type",
    "number_of_lanes",
    "speed_limit",
    "day_of_week",
    "hour",
    "weather_condition",
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "visibility",
    "nearby_intersections",
    "event_type",
    "peak_hour"
]

X = df[selected_features]
y = df["traffic_demand"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# -----------------------------------
# Train LightGBM
# -----------------------------------
from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

lgb_model.fit(X_train, y_train)

# -----------------------------------
# Train XGBoost
# -----------------------------------
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

print("✅ Both models trained successfully!")

# -----------------------------------
# Model Evaluation
# -----------------------------------
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# LightGBM Predictions
lgb_pred = lgb_model.predict(X_test)

# XGBoost Predictions
xgb_pred = xgb_model.predict(X_test)

# LightGBM Metrics
lgb_r2 = r2_score(y_test, lgb_pred)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))

# XGBoost Metrics
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print("\n===== LightGBM Results =====")
print("R² Score :", round(lgb_r2, 4))
print("MAE      :", round(lgb_mae, 4))
print("RMSE     :", round(lgb_rmse, 4))

print("\n===== XGBoost Results =====")
print("R² Score :", round(xgb_r2, 4))
print("MAE      :", round(xgb_mae, 4))
print("RMSE     :", round(xgb_rmse, 4))

import joblib

if lgb_r2 >= xgb_r2:
    best_model = lgb_model
    print("\nBest Model: LightGBM")
else:
    best_model = xgb_model
    print("\nBest Model: XGBoost")

joblib.dump(best_model, "traffic_model.pkl")
# Save feature names
joblib.dump(list(X.columns), "feature_names.pkl")

# Save the label encoders

joblib.dump(encoders, "label_encoders.pkl")

print("Model saved successfully as traffic_model.pkl")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,10))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/images/heatmap.png")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:15],
    importance["Importance"][:15]
)

plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/images/feature_importance.png")
plt.show()

best_pred = best_model.predict(X_test)

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    best_pred,
    alpha=0.5
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.savefig("static/images/actual_vs_predicted.png")
plt.show()

residuals = y_test - best_pred

plt.figure(figsize=(8,6))

plt.scatter(
    best_pred,
    residuals,
    alpha=0.5
)

plt.axhline(0, color="red")

plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.savefig("static/images/residual_plot.png")
plt.show()
