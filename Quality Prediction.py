# QUALITY PREDICTION IN MINING PLANTS
# 1. Install required libraries
!pip install xgboost scikit-learn --quiet

# 2. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 3. Upload file
from google.colab import files
uploaded = files.upload()

# 4. Load dataset
df = pd.read_csv("MiningProcess_Flotation_Plant_Database.csv")
df.columns = df.columns.str.strip()  # clean column names

# 5. Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 6. Target column
target_col = '% Silica Concentrate'
drop_cols = ['date', 'id'] if 'id' in df.columns else ['date']

# 7. Check correlation with % Iron Concentrate
corr_matrix = df.corr()
print(" Correlation between Iron and Silica:", corr_matrix['% Silica Concentrate']['% Iron Concentrate'])

# 8. Data preprocessing
features_all = df.drop(columns=[target_col])
features_wo_iron = features_all.drop(columns=['% Iron Concentrate'])
target = df[target_col]

# Impute missing values
imp = SimpleImputer(strategy='mean')
X_all = imp.fit_transform(features_all)
X_wo_iron = imp.fit_transform(features_wo_iron)
y = target.values

# Scale features
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_wo_iron_scaled = scaler.fit_transform(X_wo_iron)

# 9. Train model with and without Iron Concentrate
def evaluate_model(X, name="Model"):
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"📊 {name} Performance")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    return y_test, y_pred, model

y_true1, y_pred1, model_with_iron = evaluate_model(X_all_scaled, "With Iron Concentrate")
y_true2, y_pred2, model_wo_iron = evaluate_model(X_wo_iron_scaled, "Without Iron Concentrate")

# 10. Plot Actual vs Predicted – Extra Visualization
plt.figure(figsize=(12,4))
plt.plot(y_true1[:200], label='Actual', color='black')
plt.plot(y_pred1[:200], label='Predicted', color='darkorange')
plt.title("Actual vs Predicted % Silica (last 200 records)")
plt.ylabel("% Silica Concentrate")
plt.xlabel("Time Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Importance plot
importance = model_with_iron.feature_importances_
columns = features_all.columns
sorted_idx = np.argsort(importance)[::-1][:10]

plt.figure(figsize=(8,5))
sns.barplot(x=importance[sorted_idx], y=columns[sorted_idx], palette="viridis")
plt.title("Top 10 Important Features for % Silica Concentrate")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# 12. Future Forecasting N-hours ahead (Multistep)
# Shift target N steps back
N = 60  # Predict 60 minutes (1 hour) ahead
df_shift = df.copy()
df_shift[target_col + "_future"] = df_shift[target_col].shift(-N)
df_shift.dropna(inplace=True)

X_future = df_shift.drop(columns=[target_col, target_col + "_future"])
y_future = df_shift[target_col + "_future"]

# Prepare
Xf = imp.fit_transform(X_future)
Xf_scaled = scaler.fit_transform(Xf)
yf = y_future.values

# Train future prediction model
evaluate_model(Xf_scaled, name="1-hour Ahead Forecasting")
