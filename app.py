
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Air Quality Awareness App", layout="wide")
st.title("🌍 Air Quality Awareness & Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset1.csv")
    return df

df = load_data()
numeric_cols = ["weight", "humidity", "temperature"]
categorical_cols = ["Region", "month"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])
X = df.drop("quality", axis=1)
y = df["quality"]
model.fit(X, y)

st.header("🔍 Predict Air Quality in Your Region")
col1, col2, col3 = st.columns(3)
region = col1.selectbox("Select Region", df["Region"].unique())
month = col2.selectbox("Month", df["month"].unique())
humidity = col3.slider("Humidity", 0, 100, 40)
temperature = st.slider("Temperature (°F)", 0, 120, 70)
weight = st.slider("Weight (Dust/Particulate Level)", 500, 2000, 1000)

sample = pd.DataFrame({
    "Region": [region],
    "weight": [weight],
    "humidity": [humidity],
    "temperature": [temperature],
    "month": [month]
})
prediction = model.predict(sample)[0]
label = "Good" if prediction == 1 else "Poor"
color = "green" if label == "Good" else "red"
st.markdown(f"### 🏁 Predicted Air Quality: **:{color}[{label}]**")

st.header("📊 Region-wise Air Quality Summary")
df["predicted_quality"] = model.predict(X)
df["label"] = df["predicted_quality"].map({1: "Good", 0: "Poor"})
region_summary = df.groupby(["Region", "label"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
region_summary[["Good", "Poor"]].plot(kind="barh", stacked=True, ax=ax, color=["green", "red"])
ax.set_title("Air Quality Prediction by Region")
st.pyplot(fig)

st.header("📅 Forecast: Most Polluted Districts in 2030")

@st.cache_data
def forecast_data():
    df = pd.read_excel("Industry.xlsx")
    df['Pollutants_List'] = df['Key Air Pollutants'].str.split(', ')
    df['Health_Issues_List'] = df['Health Concerns'].str.split(', ')
    pollution_by_district = df.groupby('District')['Pollutants_List'].sum().apply(lambda x: Counter(x))
    pollution_score = pollution_by_district.apply(lambda x: sum(x.values()))
    future_years = [2026, 2027, 2028, 2029, 2030]
    future_predictions = pd.DataFrame({"District": pollution_score.index})
    for year in future_years:
        future_predictions[year] = pollution_score.values + np.random.randint(-2, 3, size=len(pollution_score))
    return df, future_predictions

df_industry, future_predictions = forecast_data()
top_2030 = future_predictions[['District', 2030]].sort_values(by=2030, ascending=False).head()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='District', y=2030, data=top_2030, palette='Reds', ax=ax)
ax.set_title('Top 5 Predicted Most Polluted Districts in 2030')
st.pyplot(fig)

st.header("🏭 Industries Affecting Public Health Most")
industry_impact = df_industry.groupby('Major_Industries')['Health_Issues_List'].sum().apply(lambda x: len(set(x)))
top_industries = industry_impact.sort_values(ascending=False).head(5)
fig, ax = plt.subplots(figsize=(8, 5))
top_industries.plot(kind='bar', color='darkred', ax=ax)
ax.set_title('Top 5 Harmful Industries')
st.pyplot(fig)
st.markdown("---")
st.caption("Built with ❤️ to raise awareness about air pollution.")
