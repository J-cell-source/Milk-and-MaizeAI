import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# APPLICATION CONFIGURATION
st.set_page_config(
    page_title="Milk and MaizeAI",
    page_icon="🌽",
    layout="wide"
)

# DATA FILES
FARMERS_FILE = "farmers.csv"
MILK_FILE = "milk_records.csv"

# CREATE DATA FILES IF THEY DO NOT EXIST
if not os.path.exists(FARMERS_FILE):
    pd.DataFrame(
        columns=["Farmer ID", "Name", "Location"]
    ).to_csv(FARMERS_FILE, index=False)

if not os.path.exists(MILK_FILE):
    pd.DataFrame(
        columns=["Farmer ID", "Date", "Litres"]
    ).to_csv(MILK_FILE, index=False)

# LOAD DATA
farmers = pd.read_csv(FARMERS_FILE)
milk = pd.read_csv(MILK_FILE)

# TITLE
st.title("🥛 Milk and MaizeAI")
st.subheader("AI-Based Agricultural Prediction System")

st.write(
    "A simple agricultural platform for recording "
    "milk production and analysing maize market prices."
)

# DASHBOARD STATISTICS
col1, col2, col3 = st.columns(3)

col1.metric("Registered Farmers", len(farmers))

col2.metric(
    "Total Milk Recorded",
    f"{milk['Litres'].sum():,.1f} L"
)

col3.metric(
    "Milk Records",
    len(milk)
)

st.divider()

# NAVIGATION
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard",
    "👨‍🌾 Register Farmer",
    "🥛 Record Milk",
    "🌽 Maize Prices",
    "🤖 AI Predictions"

])

# DASHBOARD
with tab1:

    st.header("Milk Production Overview")

    if not milk.empty:

        milk["Date"] = pd.to_datetime(milk["Date"])

        daily_milk = (
            milk.groupby("Date")["Litres"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily_milk,
            x="Date",
            y="Litres",
            markers=True,
            title="Daily Milk Production"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Milk Records")
        st.dataframe(milk.tail(10), use_container_width=True)

    else:
        st.info("No milk records available yet. Record milk to see your chart.")

    st.subheader("Registered Farmers")

    if not farmers.empty:
        st.dataframe(farmers, use_container_width=True)
    else:
        st.info("No farmers registered yet.")

# FARMER REGISTRATION
with tab2:

    st.header("Register a Farmer")

    with st.form("farmer_form"):

        name = st.text_input("Farmer's Full Name")
        location = st.text_input("Farmer's Location")

        submitted = st.form_submit_button("Register Farmer")

        if submitted:

            if name.strip() and location.strip():

                farmers = pd.read_csv(FARMERS_FILE)

                farmer_id = f"F{len(farmers) + 1:03d}"

                new_farmer = pd.DataFrame([{
                    "Farmer ID": farmer_id,
                    "Name": name.strip(),
                    "Location": location.strip()
                }])

                farmers = pd.concat(
                    [farmers, new_farmer],
                    ignore_index=True
                )

                farmers.to_csv(FARMERS_FILE, index=False)

                st.success(
                    f"Farmer registered successfully! ID: {farmer_id}"
                )

                st.rerun()

            else:
                st.error("Please enter the farmer's name and location.")

# MILK RECORDING
with tab3:

    st.header("Record Daily Milk Production")

    farmers = pd.read_csv(FARMERS_FILE)

    if not farmers.empty:

        farmer_ids = farmers["Farmer ID"].tolist()

        farmer_id = st.selectbox(
            "Select Farmer",
            farmer_ids,
            format_func=lambda x: (
                farmers.loc[
                    farmers["Farmer ID"] == x, "Name"
                ].iloc[0]
            )
        )

        record_date = st.date_input("Recording Date")

        litres = st.number_input(
            "Milk Produced (Litres)",
            min_value=0.1,
            step=0.5
        )

        if st.button("Save Milk Record"):

            milk = pd.read_csv(MILK_FILE)

            new_record = pd.DataFrame([{
                "Farmer ID": farmer_id,
                "Date": str(record_date),
                "Litres": litres
            }])

            milk = pd.concat(
                [milk, new_record],
                ignore_index=True
            )

            milk.to_csv(MILK_FILE, index=False)

            st.success("Milk production record saved successfully!")

            st.rerun()

    else:
        st.warning("Please register a farmer before recording milk.")
# MAIZE PRICE ANALYSIS
with tab4:

    st.header("Maize Market Price Analysis")

    maize = pd.read_csv("maize_prices.csv")

    maize["Date"] = pd.to_datetime(maize["Date"])

    st.subheader("Maize Prices by Market")

    fig = px.line(
        maize,
        x="Date",
        y="Price",
        color="Market",
        markers=True,
        title="Maize Price Trends"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Market Prices")

    latest = (
        maize.sort_values("Date")
        .groupby("Market")
        .tail(1)
    )

    st.dataframe(latest, use_container_width=True)

    best_market = latest.loc[
        latest["Price"].idxmax()
    ]

    st.success(
        f"Highest estimated selling price: "
        f"{best_market['Market']} - "
        f"KSh {best_market['Price']:,.0f}"
    )
# AI PREDICTION PAGE
with tab5:

    st.header("🤖 Milk and MaizeAI Predictions")

    st.write(
        "Use historical agricultural data to generate "
        "milk production and maize price forecasts."
    )

    # MILK PREDICTION
    st.subheader("🥛 Milk Production Forecast")

    if os.path.exists("milk_model.pkl"):

        milk_data = pd.read_csv("milk_records.csv")
        farmers_data = pd.read_csv("farmers.csv")

        if not milk_data.empty and not farmers_data.empty:

            farmer_ids = farmers_data["Farmer ID"].tolist()

            selected_farmer = st.selectbox(
                "Select Farmer for Prediction",
                farmer_ids,
                format_func=lambda x: farmers_data.loc[
                    farmers_data["Farmer ID"] == x,
                    "Name"
                ].iloc[0],
                key="prediction_farmer"
            )

            farmer_history = milk_data[
                milk_data["Farmer ID"] == selected_farmer
            ].copy()

            farmer_history["Date"] = pd.to_datetime(
                farmer_history["Date"]
            )

            farmer_history = farmer_history.sort_values("Date")

            if len(farmer_history) >= 2:

                model = joblib.load("milk_model.pkl")

                previous = farmer_history["Litres"].astype(float).tolist()

                forecast = []

                for day in range(1, 8):

                    features = pd.DataFrame([{
                        "Previous_Day": previous[-1],
                        "Two_Days_Before": previous[-2]
                    }])

                    prediction = float(model.predict(features)[0])

                    prediction = max(0, prediction)

                    forecast.append({
                        "Day": day,
                        "Predicted Milk (Litres)": round(prediction, 2)
                    })

                    previous.append(prediction)

                forecast_df = pd.DataFrame(forecast)

                st.success(
                    "Milk production forecast generated successfully."
                )

                st.dataframe(
                    forecast_df,
                    use_container_width=True
                )

                fig = px.line(
                    forecast_df,
                    x="Day",
                    y="Predicted Milk (Litres)",
                    markers=True,
                    title="Next 7 Days Milk Production Forecast"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.warning(
                    "This farmer needs at least two milk records "
                    "before prediction."
                )

        else:
            st.warning("Register a farmer and record milk first.")

    else:
        st.warning(
            "Milk model not found. Run: python milk_model.py"
        )

    st.divider()

    # MAIZE PREDICTION
    st.subheader("🌽 Maize Price Forecast")

    if os.path.exists("maize_models.pkl"):

        maize_data = pd.read_csv("maize_prices.csv")
        maize_data["Date"] = pd.to_datetime(maize_data["Date"])

        models = joblib.load("maize_models.pkl")

        predictions = []

        for market, details in models.items():

            model = details["model"]
            start_date = pd.to_datetime(details["start_date"])

            market_history = maize_data[
                maize_data["Market"] == market
            ].copy()

            last_date = market_history["Date"].max()

            for day in range(1, 8):

                future_date = last_date + pd.Timedelta(days=day)

                day_number = (future_date - start_date).days

                predicted_price = float(
                    model.predict(
                        pd.DataFrame({"Day": [day_number]})
                    )[0]
                )

                predictions.append({
                    "Market": market,
                    "Date": future_date,
                    "Predicted Price (KSh)": round(
                        max(0, predicted_price), 2
                    )
                })

        if predictions:

            prediction_df = pd.DataFrame(predictions)

            st.success(
                "Maize price forecasts generated successfully."
            )

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

            fig = px.line(
                prediction_df,
                x="Date",
                y="Predicted Price (KSh)",
                color="Market",
                markers=True,
                title="Next 7 Days Maize Price Forecast"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            latest_forecasts = (
                prediction_df.sort_values("Date")
                .groupby("Market")
                .tail(1)
            )

            best = latest_forecasts.loc[
                latest_forecasts[
                    "Predicted Price (KSh)"
                ].idxmax()
            ]

            st.info(
                f"Highest predicted selling price: "
                f"{best['Market']} – "
                f"KSh {best['Predicted Price (KSh)']:,.2f}"
            )

        else:
            st.warning("No maize prediction models available.")

    else:
        st.warning(
            "Maize model not found. Run: python maize_model.py"
        )
