import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# LOAD MILK DATA
milk = pd.read_csv("milk_records.csv")

if len(milk) < 5:
    raise ValueError(
        "At least 5 milk records are required for this demonstration."
    )

milk["Date"] = pd.to_datetime(milk["Date"])
milk = milk.sort_values(["Farmer ID", "Date"])

# CREATE PREVIOUS-DAY FEATURES
milk["Previous_Day"] = (
    milk.groupby("Farmer ID")["Litres"].shift(1)
)

milk["Two_Days_Before"] = (
    milk.groupby("Farmer ID")["Litres"].shift(2)
)

data = milk.dropna().copy()

if len(data) < 4:
    raise ValueError(
        "Not enough historical records to train the model."
    )

X = data[["Previous_Day", "Two_Days_Before"]]
y = data["Litres"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# TRAIN AI MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# EVALUATE MODEL
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Milk Prediction Model Trained!")
print("Mean Absolute Error:", round(mae, 2), "litres")

# SAVE MODEL
joblib.dump(model, "milk_model.pkl")

print("Model saved successfully!")

