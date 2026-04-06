import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD DATA ----------------
df = pd.read_csv("earthquake_1995-2023.csv")

print(df.head())
print(df.info())

# ---------------- DATA CLEANING ----------------

# Drop useless columns
df = df.drop(['title', 'location'], axis=1)

# Fill missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert date_time
df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)

# Extract features
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day

df = df.drop(['date_time'], axis=1)

print(df.head())
print(df.info())

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(8,5))
plt.scatter(df['longitude'], df['latitude'], alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Earthquake Locations")
plt.show()

# ---------------- FEATURE ENGINEERING ----------------

df = pd.get_dummies(df, columns=['alert', 'net', 'magType', 'continent', 'country'])

X = df.drop(['magnitude'], axis=1)
y = df['magnitude']

print(X.shape)
print(y.shape)

# ---------------- TRAIN TEST SPLIT ----------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))

# Neural Network
nn = MLPRegressor(hidden_layer_sizes=(16,16), max_iter=1000)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)

print("Neural Network MAE:", mean_absolute_error(y_test, y_pred_nn))

# ---------------- GRID SEARCH ----------------

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# Use best model
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("Tuned Random Forest MAE:", mean_absolute_error(y_test, y_pred_best))

# ---------------- SAVE MODEL ----------------

import joblib
joblib.dump(rf, "earthquake_model.pkl")

# ---------------- PREDICTION FUNCTION ----------------

def predict_earthquake(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = rf.predict(input_df)
    return prediction[0]

# ---------------- SAMPLE TEST ----------------

sample = {
    'cdi': 7,
    'mmi': 5,
    'tsunami': 0,
    'sig': 800,
    'nst': 100,
    'dmin': 0.5,
    'gap': 30,
    'depth': 100,
    'latitude': 10,
    'longitude': 70,
    'year': 2023,
    'month': 5,
    'day': 10,
    'alert_green': 1,
    'net_us': 1,
    'magType_mb': 1,
    'continent_Asia': 1,
    'country_India': 1
}

print("Sample Prediction:", predict_earthquake(sample))

# ---------------- USER INPUT ----------------

def get_user_input():
    data = {}

    data['cdi'] = int(input("Enter CDI: "))
    data['mmi'] = int(input("Enter MMI: "))
    data['tsunami'] = int(input("Tsunami (0 or 1): "))
    data['sig'] = int(input("Enter significance: "))
    data['nst'] = int(input("Enter NST: "))
    data['dmin'] = float(input("Enter dmin: "))
    data['gap'] = float(input("Enter gap: "))
    data['depth'] = float(input("Enter depth: "))
    data['latitude'] = float(input("Enter latitude: "))
    data['longitude'] = float(input("Enter longitude: "))
    data['year'] = int(input("Enter year: "))
    data['month'] = int(input("Enter month: "))
    data['day'] = int(input("Enter day: "))

    data['alert_green'] = 1
    data['net_us'] = 1
    data['magType_mb'] = 1
    data['continent_Asia'] = 1
    data['country_India'] = 1

    return data

# Run only when needed
if __name__ == "__main__":
    user_data = get_user_input()
    print("Predicted Magnitude:", predict_earthquake(user_data))