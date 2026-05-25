import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="California Housing Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00F5FF;
}

.stButton>button {
    background-color: #00F5FF;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🏠 California Housing Price Prediction")

st.markdown("---")

# ---------------- LOAD DATA ----------------

data = fetch_california_housing()

X = data.data
y = data.target

features = data.feature_names

df = pd.DataFrame(X, columns=features)

df['Price'] = y

# ---------------- DATA PREVIEW ----------------

st.subheader("📊 Dataset Preview")

st.dataframe(df.head())

# ---------------- OUTLIER REMOVAL ----------------

cleaned_df = df.copy()

for i in cleaned_df.columns[:-1]:

    Q1 = cleaned_df[i].quantile(0.25)
    Q3 = cleaned_df[i].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    cleaned_df = cleaned_df[
        (cleaned_df[i] >= lower_limit) &
        (cleaned_df[i] <= upper_limit)
    ]

# ---------------- SIDE BY SIDE PLOTS ----------------

st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

# -------- BOX PLOT --------

with col1:

    selected_column = st.selectbox(
        "Select Feature",
        cleaned_df.columns[:-1]
    )

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.boxplot(cleaned_df[selected_column])

    ax1.set_title(f"{selected_column} Boxplot")

    st.pyplot(fig1)

# -------- SCATTER PLOT --------

with col2:

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.scatter(
        cleaned_df['MedInc'],
        cleaned_df['Price'],
        alpha=0.5
    )

    ax2.set_xlabel("Median Income")

    ax2.set_ylabel("House Price")

    ax2.set_title("Income vs Price")

    st.pyplot(fig2)

# ---------------- FEATURES AND TARGET ----------------

X = cleaned_df.drop('Price', axis=1)

y = cleaned_df['Price']

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

# ---------------- PERFORMANCE ----------------

st.subheader("📊 Model Performance")

m1, m2 = st.columns(2)

with m1:
    st.metric("R² Score", round(r2, 4))

with m2:
    st.metric("MSE", round(mse, 4))

# ---------------- ACTUAL VS PREDICTED ----------------

st.subheader("📉 Actual vs Predicted")

fig3, ax3 = plt.subplots(figsize=(10,5))

ax3.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r'
)

ax3.set_xlabel("Actual Prices")

ax3.set_ylabel("Predicted Prices")

ax3.set_title("Linear Regression Predictions")

st.pyplot(fig3)

# ---------------- USER INPUT SECTION ----------------

st.markdown("---")

st.subheader("🏡 Predict House Price")

c1, c2 = st.columns(2)

with c1:

    MedInc = st.slider(
        "Median Income",
        0.0,
        15.0,
        3.5
    )

    HouseAge = st.slider(
        "House Age",
        1.0,
        60.0,
        25.0
    )

    AveRooms = st.slider(
        "Average Rooms",
        1.0,
        15.0,
        5.0
    )

    AveBedrms = st.slider(
        "Average Bedrooms",
        1.0,
        5.0,
        1.0
    )

with c2:

    Population = st.slider(
        "Population",
        1.0,
        10000.0,
        1000.0
    )

    AveOccup = st.slider(
        "Average Occupancy",
        1.0,
        10.0,
        3.0
    )

    Latitude = st.slider(
        "Latitude",
        32.0,
        42.0,
        34.0
    )

    Longitude = st.slider(
        "Longitude",
        -125.0,
        -114.0,
        -118.0
    )

# ---------------- PREDICTION ----------------

input_data = np.array([[
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
]])

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

# ---------------- BUTTON ----------------

if st.button("Predict Price"):

    st.success(
        f"🏠 Predicted House Price: ${prediction[0] * 100000:.2f}"
    )
