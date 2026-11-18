import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# LOAD MAIZE DATA
maize = pd.read_csv("maize_prices.csv")

maize["Date"] = pd.to_datetime(maize["Date"])

# TRAIN ONE MODEL FOR EACH MARKET
models = {}
errors = {}

for market in maize["Market"].unique():

    market_data = maize[
        maize["Market"] == market
    ].sort_values("Date").copy()

    # Convert dates into sequential day numbers
    market_data["Day"] = (
        market_data["Date"] - market_data["Date"].min()
    ).dt.days

    X = market_data[["Day"]]
    y = market_data["Price"]

    if len(market_data) < 5:
        print(f"Not enough data for {market}")
        continue

    # Use earlier records for training and latest records for testing
    split = len(market_data) - 2

    model = LinearRegression()

    model.fit(X.iloc[:split], y.iloc[:split])

    test_predictions = model.predict(X.iloc[split:])

    mae = mean_absolute_error(
        y.iloc[split:],
        test_predictions
    )

    # Retrain using all available records
    model.fit(X, y)

    models[market] = {
        "model": model,
        "start_date": market_data["Date"].min()
    }

    errors[market] = mae

# SAVE MODELS
joblib.dump(models, "maize_models.pkl")

print("Maize prediction models trained successfully!")

for market, error in errors.items():
    print(f"{market} MAE: KSh {error:.2f}")

print("Models saved in maize_models.pkl")

