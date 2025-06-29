import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 🎯 Set page config
st.set_page_config(
    page_title="COVID-19 Recovery Predictor",
    layout="wide"
)

# 🏷️ Title
st.title("🦠 Health vs. Pandemic: COVID-19 Recovery Predictor")

# 📥 Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    return df

df = load_data()

# 📊 Show raw data
st.subheader("📋 Preview of Data")
st.dataframe(df.head())

# 📈 Feature selection
st.subheader("🌍 Select a Country to Predict Recovery Rate")

# Dropdown to select country
country_list = sorted(df['country'].unique())
selected_country = st.selectbox("Select Country", country_list)

# Filter the selected country's data
country_data = df[df['country'] == selected_country]

if not country_data.empty:
    st.success(f"✅ Data for {selected_country} loaded.")

    # Features for prediction
    features = [
        'gdp_per_capita',
        'health_expenditure_percent_gdp',
        'people_fully_vaccinated',
        'total_tests',
        'population_x',
        'population_density',
        'life_expectancy'
    ]

    X = df[features]
    y = df['recovery_rate']

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict recovery rate for selected country
    input_data = country_data[features].iloc[0].values.reshape(1, -1)
    predicted = model.predict(input_data)[0]

    actual = country_data['recovery_rate'].values[0]

    # 📊 Display Prediction
    st.metric(label="📈 Predicted Recovery Rate (%)", value=f"{predicted:.2f}")
    st.metric(label="✅ Actual Recovery Rate (%)", value=f"{actual:.2f}")

    # Bar comparison chart
    st.subheader("🔍 Actual vs Predicted")
    comparison_df = pd.DataFrame({
        'Recovery Rate': ['Actual', 'Predicted'],
        'Percentage': [actual, predicted]
    })

    fig, ax = plt.subplots()
    bars = ax.bar(comparison_df['Recovery Rate'], comparison_df['Percentage'], color=["skyblue", "orange"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.set_ylabel("Recovery Rate (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

else:
    st.warning("No data available for this country.")

st.markdown("---")
st.caption("📍 Created by Jhalak | Powered by Streamlit, Pandas, Scikit-learn")
