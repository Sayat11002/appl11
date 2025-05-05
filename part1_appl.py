import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Page Config ===
st.set_page_config(page_title="🚘 Car Assistant", page_icon="🧠", layout="wide")

# === THEME SWITCHER ===
theme = st.radio("🌗 Choose Theme", ["Light mode", "Dark mode"], horizontal=True)

if theme == "Light mode":
    background_color = "#f5f0e6"
    text_color = "#1f1f1f"
    button_color = "#e0dbd1"
    hover_color = "#d1cfc7"
else:
    background_color = "#1e1e1e"
    text_color = "#808000"
    button_color = "#333333"
    hover_color = "#444444"

# === CSS Style ===
st.markdown(f"""<style>
html, body, [class*="css"], .main, .block-container {{
    background-color: {background_color} !important;
    color: {text_color} !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}}
h1, h2, h3, h4, h5, h6, p, span, label, div {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}
.stButton > button {{
    background-color: {button_color};
    color: {text_color};
    border: 1px solid #888;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    transition: all 0.2s ease-in-out;
    font-size: 16px;
    font-weight: 600;
}}
.stButton > button:hover {{
    background-color: {hover_color};
    transform: scale(1.05);
}}
input, textarea, select {{
    background-color: {button_color} !important;
    color: {text_color} !important;
    border: 1px solid #aaa !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}}
::placeholder {{
    color: {text_color}99 !important;
    font-size: 15px !important;
}} </style>
""", unsafe_allow_html=True)

# === Load & preprocess ===
@st.cache_data
def load_price_data():
    df = pd.read_csv("22613data.csv")
    df = df.drop(['City', 'Volume'], axis=1)
    return df

@st.cache_data
def remove_outliers(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

@st.cache_data
def preprocess_data(data):
    df = data.drop_duplicates()
    df.fillna({'Mark': 'Unknown', 'Fuel Type': 'Unknown', 'Transmission': 'Unknown'}, inplace=True)
    df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())
    df = remove_outliers(df, 'Price')
    df = remove_outliers(df, 'Mileage')
    encoders = {}
    for col in ['Company', 'Mark', 'Fuel Type', 'Transmission', 'Car_type']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.upper())
        encoders[col] = le
    return df, encoders

@st.cache_resource
def train_price_model(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=350, random_state=4)
    model.fit(X, y)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("22613data.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_odometer_model(df):
    df = df.dropna(subset=["Year", "Volume", "Mileage", "Mark"])
    group_median = df.groupby(["Year", "Mark", "Volume"])["Mileage"].median().reset_index().rename(columns={"Mileage": "MedianMileage"})
    df = df.merge(group_median, on=["Year", "Mark", "Volume"], how="left")
    df["OdometerNormal"] = (df["Mileage"] >= 0.65 * df["MedianMileage"]).astype(int)

    le_dict = {}
    for col in ["Mark", "Fuel Type", "Transmission", "Car_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    features = ["Year", "Volume", "Mileage", "Mark", "Fuel Type", "Transmission", "Car_type"]
    X = df[features]
    y = df["OdometerNormal"]
    return X, y, le_dict

@st.cache_data
def train_odometer_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# === Load & Train ===
raw_data = load_price_data()
df, encoders = preprocess_data(raw_data)
model = train_price_model(df)

# === Interface ===
st.title("🚘 Car Assistant")
tabs = st.tabs(["Car find", "💰 Estimate Price", "📆 Credit Calc"])

# === Tab 1 ===
with tabs[0]:
    full_df = load_data()
    st.header("📊 Popularity & 🔎 Mileage Consistency")
    tab_choice = st.radio("Select View",["Company Popularity","Brand Popularity", "Odometer Checker"], horizontal=True)

    if tab_choice == "Company Popularity":
        city = st.selectbox("Select City", sorted(full_df["City"].unique()))
        top_brands = full_df[full_df["City"] == city]["Company"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax)
        ax.set_title(f"Top 5 Brands in {city}")
        st.pyplot(fig)

    elif tab_choice == "Brand Popularity":
        company = st.selectbox("Select Company", sorted(full_df["Company"].unique()))
        brand_count = full_df[full_df["Company"] == company]["Mark"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x=brand_count.values, y=brand_count.index, ax=ax)
        ax.set_title(f"Top 5 Models of {company}")
        st.pyplot(fig)

    elif tab_choice == "Odometer Checker":
        X_odo, y_odo, le_dict = preprocess_odometer_model(full_df)
        model_odo = train_odometer_model(X_odo, y_odo)

        st.markdown("### Input Car Details:")
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox("Company", full_df["Company"].unique())
            mark = st.selectbox("Mark", full_df[full_df["Company"] == company]["Mark"].unique())
            year = st.number_input("Year", 2000, 2025, 2015)
            mileage = st.number_input("Mileage (km)", value=100000)
        with col2:
            volume = st.number_input("Engine Volume (L)", value=2.0)
            fuel = st.selectbox("Fuel Type", full_df["Fuel Type"].unique())
            trans = st.selectbox("Transmission", full_df["Transmission"].unique())
            ctype = st.selectbox("Car Type", full_df["Car_type"].unique())

        if st.button("Check Odometer Integrity"):
            input_data = pd.DataFrame([{
                "Year": year,
                "Volume": volume,
                "Mileage": mileage,
                "Mark": le_dict["Mark"].transform([mark])[0],
                "Fuel Type": le_dict["Fuel Type"].transform([fuel])[0],
                "Transmission": le_dict["Transmission"].transform([trans])[0],
                "Car_type": le_dict["Car_type"].transform([ctype])[0],
            }])
            pred = model_odo.predict(input_data)[0]
            if pred == 1:
                st.success("✅ Odometer reading appears NORMAL")
            else:
                st.warning("⚠️ Possible MILEAGE TAMPERING detected")

# === Tab 2: Estimate Price ===
with tabs[1]:
    st.subheader("💰 Estimate Car Price")
    company = st.selectbox("Company", raw_data["Company"].unique(), key="price_company")
    mark = st.selectbox("Mark", raw_data[raw_data["Company"] == company]["Mark"].unique(), key="price_mark")
    year = st.number_input("Year", 2000, 2025, 2015, key="price_year")
    mileage = st.number_input("Mileage (km)", value=100000, key="price_mileage")
    fuel = st.selectbox("Fuel Type", raw_data["Fuel Type"].unique(), key="price_fuel")
    trans = st.selectbox("Transmission", raw_data["Transmission"].unique(), key="price_trans")
    ctype = st.selectbox("Car Type", raw_data["Car_type"].unique(), key="price_type")

    if st.button("📈 Estimate Price"):
        try:
            new_car = pd.DataFrame({
                "Company": [company],
                "Mark": [mark],
                "Year": [year],
                "Fuel Type": [fuel],
                "Transmission": [trans],
                "Mileage": [mileage],
                "Car_type": [ctype]
            })
            for col in new_car.columns:
                if col in encoders:
                    new_car[col] = new_car[col].astype(str).str.upper()
                    new_car[col] = encoders[col].transform(new_car[col])

            prediction = model.predict(new_car)[0]
            st.success(f"💵 Estimated Price: **{int(prediction):,} ₸**")

            st.markdown("### 🔍 Similar Listings:")
            similar = raw_data[
                (raw_data["Company"] == company) &
                (raw_data["Mark"] == mark) &
                (raw_data["Fuel Type"] == fuel) &
                (raw_data["Transmission"] == trans) &
                (raw_data["Car_type"] == ctype)
            ].copy()
            similar["Mileage_Diff"] = abs(similar["Mileage"] - mileage)
            similar["Year_Diff"] = abs(similar["Year"] - year)
            similar["Score"] = similar["Mileage_Diff"] + similar["Year_Diff"] * 500
            top3 = similar.sort_values("Score").head(3)
            st.dataframe(top3[["Company", "Mark", "Year", "Fuel Type", "Transmission", "Mileage", "Car_type", "City"]])

        except Exception as e:
            st.error(f"Error: {e}")

# === Tab 3: Credit Calculator ===
with tabs[2]:
    st.subheader("📆 Credit Calculator")
    car_price = st.number_input("Car Price (₸)", min_value=100000, value=1000000, step=10000)
    down_payment = st.number_input("Down Payment (₸)", min_value=0, max_value=car_price, value=int(car_price * 0.2), step=10000)
    term = st.slider("Term (months)", 6, 84, 36, step=6)
    rate = st.slider("Interest Rate (%/yr)", 0.0, 100.0, 10.0, step=0.1)

    if car_price > down_payment:
        loan = car_price - down_payment
        monthly_rate = (rate / 100) / 12
        if rate > 0:
            monthly = loan * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        else:
            monthly = loan / term
        st.success(f"Monthly payment: **{int(monthly):,} ₸**")
    else:
        st.warning("Down payment should be less than car price.")
