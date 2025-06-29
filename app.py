import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import io

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="COVID-19 Recovery Predictor", layout="wide")
st.title("🦠 Health vs. Pandemic: COVID-19 Recovery Rate Predictor")
st.markdown("A Streamlit app to predict COVID-19 recovery rates using health infrastructure data.")

# ------------------------------
# Load Data
# ------------------------------
st.header("📊 Dataset Preview")
try:
    df = pd.read_csv('data/cleaned_data.csv')
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ CSV file not found! Make sure 'data/cleaned_data.csv' exists in your repo.")
    st.stop()

# ------------------------------
# Correlation Heatmap
# ------------------------------
st.header("📈 Feature Correlation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ------------------------------
# Train Model
# ------------------------------
st.header("🧠 Train Recovery Rate Prediction Model")

# Select features (you can adjust column names as per your dataset)
feature_cols = ['hospital_beds_per_1000', 'gdp_per_capita', 'doctors_per_1000']
target_col = 'recovery_rate'

if not all(col in df.columns for col in feature_cols + [target_col]):
    st.error("Required columns not found in CSV!")
    st.stop()

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Input Prediction
# ------------------------------
st.subheader("🔍 Predict Recovery Rate")

col1, col2, col3 = st.columns(3)

with col1:
    beds = st.number_input("Hospital beds per 1000", min_value=0.0, max_value=20.0, value=5.0)
with col2:
    gdp = st.number_input("GDP per capita", min_value=0.0, max_value=100000.0, value=10000.0)
with col3:
    doctors = st.number_input("Doctors per 1000", min_value=0.0, max_value=10.0, value=1.5)

input_df = pd.DataFrame([[beds, gdp, doctors]], columns=feature_cols)
predicted = model.predict(input_df)[0]
st.success(f"✅ Predicted Recovery Rate: **{predicted:.2f}%**")

# ------------------------------
# Export Prediction to CSV
# ------------------------------
st.subheader("⬇️ Export Prediction Result")

result_df = pd.DataFrame({
    'Hospital Beds per 1000': [beds],
    'GDP per Capita': [gdp],
    'Doctors per 1000': [doctors],
    'Predicted Recovery Rate (%)': [round(predicted, 2)]
})

csv_data = result_df.to_csv(index=False)
st.download_button(
    label="Download Result as CSV",
    data=csv_data,
    file_name='prediction_result.csv',
    mime='text/csv'
)
