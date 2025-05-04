import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
# === Page Config ===
st.set_page_config(
    page_title="🚘 Car Assistant",
    page_icon="🧠",
    layout="wide"
)

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

# === Dynamic CSS based on theme ===
st.markdown(f""" <style>
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

# === Price Estimation Functions ==
@st.cache_data
def load_price_data():
    car_df = pd.read_csv("22613data.csv")
    car_df = car_df.drop(['City', 'Volume'], axis=1)
    return car_df
raw_data = load_price_data()
categorical_cols = ['Company', 'Mark', 'Fuel Type', 'Transmission', 'Car_type']
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
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.upper())
        encoders[col] = le

    return df, encoders

df, encoders = preprocess_data(raw_data)

@st.cache_resource
def train_model(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=350, random_state=4)
    model.fit(X, y)
    return model

model = train_model(df)
@st.cache_data
def load_data():
    df = pd.read_csv("22613data.csv")
    df.columns = df.columns.str.strip()  
    return df
@st.cache_data
def preprocess_odometer_model(df):
    df = df.copy()
    df = df.dropna(subset=["Year", "Volume", "Mileage", "Mark"])
    group_median = (
        df.groupby(["Year", "Mark", "Volume"])["Mileage"]
        .median()
        .reset_index()
        .rename(columns={"Mileage": "MedianMileage"})
    )

    df = df.merge(group_median, on=["Year", "Mark", "Volume"], how="left")
    df["OdometerNormal"] = (df["Mileage"] >= 0.65 * df["MedianMileage"]).astype(int)  # 1 - нормальный, 0 - занижен

    cat_cols = ["Mark", "Fuel Type", "Transmission", "Car_type"]
    le_dict = {}
    for col in cat_cols:
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
# === Interface ===
st.title("🚘 Car Assistant")

tabs = st.tabs(["Car find","💰 Estimate Price", "📆 Credit Calc"])
with tabs[0]:
    df = load_data()  
    st.header("📊 Popularity & 🔎 Mileage Consistency")
    tab_choice = st.radio(["Company Popularity","Brand Popularity", "Odometer Checker"], horizontal=True)
    if tab_choice == "Company Popularity":
        st.subheader("Popular Car Brands by City")
        selected_city = st.selectbox("Select City", sorted(df["City"].unique()))
        filtered_df = df[df["City"] == selected_city]
        top_brands = filtered_df["Company"].value_counts().head(5)
        fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(10,7))
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax)
        ax.set_xlabel("Number of Listings")
        ax.set_ylabel("Brand")
        ax.set_title(f"Top 5 Brands in {selected_city.title()}")
        st.pyplot(fig)
    elif tab_choice == "Brand Popularity":
        st.subheader("Popular Car Brands by Company and Mark")
        selected_company = st.selectbox("Select Company", sorted(df["Company"].unique()))
        filtered_df = df[df["Company"] == selected_company]
        # Группируем сначала по компании, потом по марке
        brand_popularity = filtered_df.groupby("Mark")["Company"].count().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(10,7))
        sns.barplot(x=brand_popularity.values, y=brand_popularity.index, ax=ax)
        ax.set_xlabel("Number of Listings")
        ax.set_ylabel("Brand")
        ax.set_title(f"Top 5 Brands of {selected_company} Company")
        st.pyplot(fig)
    # Вкладка "Odometer Checker"
    elif tab_choice == "Odometer Checker":
        st.subheader("🔍 Detect Suspected Odometer Rollback")
        X, y, le_dict = preprocess_odometer_model(df)
        model = train_odometer_model(X, y)
        st.markdown("### Input Car Details:")
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox("Company", df["Company"].unique())
            mark = st.selectbox("Mark", df[df["Company"] == company]["Mark"].unique())
            year = st.number_input("Year", min_value=2000, max_value=2025, value=2015)
            mileage = st.number_input("Mileage (km)", value=100000)

        with col2:
            volume = st.number_input("Engine Volume (L)", value=2.0)  
            fuel = st.selectbox("Fuel Type", df["Fuel Type"].unique())
            trans = st.selectbox("Transmission", df["Transmission"].unique())
            ctype = st.selectbox("Car Type", df["Car_type"].unique())

        if st.button("Check Odometer Integrity"):
            input_dict = {
                "Year": year,
                "Volume": volume, 
                "Mileage": mileage,
                "Mark": le_dict["Mark"].transform([mark])[0],
                "Fuel Type": le_dict["Fuel Type"].transform([fuel])[0],
                "Transmission": le_dict["Transmission"].transform([trans])[0],
                "Car_type": le_dict["Car_type"].transform([ctype])[0],
            }

            X_input = pd.DataFrame([input_dict])
            prob = model.predict_proba(X_input)[0][1]
            pred = model.predict(X_input)[0]

            if pred == 1:
                st.success(f"✅ Odometer reading appears NORMAL)")
            else:
                st.warning(f"⚠️ Possible MILEAGE TAMPERING detected)")

# === Tab 1: Estimate Price ===
with tabs[1]:
    st.markdown("### 📊 Enter car details to estimate the price:")
    company = st.selectbox("🏢 Manufacturer", sorted(raw_data['Company'].dropna().unique()), key="company_select")
    filtered_data = raw_data[raw_data['Company'] == company]
    mark = st.selectbox("🚘 Model", sorted(filtered_data['Mark'].dropna().unique()), key="model_select")
    year = st.number_input("📅 Year", 1990, 2025, 2015, key="year_input")
    fuel = st.selectbox("⛽ Fuel Type", sorted(raw_data['Fuel Type'].dropna().unique()), key="fuel_select")
    trans = st.selectbox("⚙️ Transmission", sorted(raw_data['Transmission'].dropna().unique()), key="trans_select")
    mileage = st.number_input("🛣️ Mileage (km)", 0, 1_000_000, 100_000, key="mileage_input")
    car_type = st.selectbox("🚗 Body Type", sorted(raw_data['Car_type'].dropna().unique()), key="type_select")
    if st.button("📈 Estimate Price", key="price_button"):
        new_car = pd.DataFrame({
            'Company': [company],
            'Mark': [mark],
            'Year': [year],
            'Fuel Type': [fuel],
            'Transmission': [trans],
            'Mileage': [mileage],
            'Car_type': [car_type]
        })

        try:
            for col in categorical_cols:
                new_car[col] = new_car[col].astype(str).str.upper()
                if any(v not in encoders[col].classes_ for v in new_car[col]):
                    raise ValueError(f"❌ Unknown value in column '{col}'")
                new_car[col] = encoders[col].transform(new_car[col])

            pred = model.predict(new_car)[0]
            st.success(f"💵 Estimated Price: **{int(pred):,} ₸**")
        except ValueError as e:
            st.error(str(e))

# === Tab 2: Credit Calculator ===
with tabs[2]:
    st.markdown("### 📆 Credit calculator coming soon...")
    car_price = st.number_input("Car Price (₸)", min_value=100000, value=1000000, step=10000, key="price_input")
    down_payment = st.number_input("Down Payment (₸)", min_value=0, max_value=car_price, value=int(car_price * 0.2), step=10000, key="down_payment")
    term = st.slider("Term (months)", 6, 84, 36, step=6, key="term_slider")
    rate = st.slider("Interest (%/yr)", 0.0, 100.0, 10.0, step=0.1, key="rate_slider")

    if car_price > down_payment:
        loan = car_price - down_payment
        monthly_rate = (rate / 100) / 12

        if rate > 0:
            m = monthly_rate
            monthly = loan * (m * (1 + m)**term) / ((1 + m)**term - 1)
        else:
            monthly = loan / term

        st.success(f"Monthly: **{int(monthly):,} ₸**")
    else:
        st.warning("Down payment >= price")

