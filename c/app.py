# Smart City Assistant - Streamlit App (Updated)
# Filename: app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Sustainable Smart City Assistant", layout="wide")
st.title("🌆 Sustainable Smart City Assistant")
st.markdown("""
This assistant helps city administrators, planners, and citizens with insights, feedback, forecasts, and sustainability tips.
""")

menu = st.sidebar.selectbox("Choose a module", [
    "📄 Policy Search & Summarization",
    "👥 Citizen Feedback",
    "📊 KPI Forecasting",
    "🌍 Eco Tips Generator",
    "⚡ Anomaly Detection",
    "🤖 Chat Assistant"
])

# Load IBM Granite LLM locally
model_path = "ibm-granite/granite-3.3-2b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def call_granite(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

if menu == "📄 Policy Search & Summarization":
    st.header("Policy Search & Summarization")
    uploaded_file = st.file_uploader("Upload a policy document", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        if st.button("Summarize"):
            with st.spinner("Summarizing policy..."):
                prompt = f"Summarize the following policy document for a general audience:\n{text}"
                summary = call_granite(prompt)
                st.subheader("📖 Summary")
                st.write(summary)

elif menu == "👥 Citizen Feedback":
    st.header("Citizen Feedback Form")
    name = st.text_input("Your Name")
    category = st.selectbox("Issue Category", ["Water", "Electricity", "Road", "Garbage", "Other"])
    location = st.text_input("Location")
    description = st.text_area("Describe the issue")
    if st.button("Submit Feedback"):
        feedback = {
            "name": name,
            "category": category,
            "location": location,
            "description": description
        }
        st.success("Feedback submitted successfully!")
        st.json(feedback)

elif menu == "📊 KPI Forecasting":
    st.header("KPI Forecasting")
    kpi_file = st.file_uploader("Upload last year's KPI data (CSV)", type="csv")
    if kpi_file:
        df = pd.read_csv(kpi_file)
        st.write("Uploaded Data:", df.head())
        if "Month" in df.columns and "Value" in df.columns:
            df["MonthIndex"] = np.arange(len(df))
            reg = LinearRegression()
            reg.fit(df[["MonthIndex"]], df["Value"])
            next_index = [[len(df)]]
            forecast = reg.predict(next_index)[0]
            df_forecast = df.copy()
            df_forecast.loc[len(df)] = ["Next Month", forecast, next_index[0][0]]
            fig = px.line(df_forecast, x="Month", y="Value", title="KPI Forecast")
            st.subheader("🌐 Forecasted Results")
            st.write(df_forecast)
            st.plotly_chart(fig)
        else:
            st.error("CSV must contain 'Month' and 'Value' columns.")

elif menu == "🌍 Eco Tips Generator":
    st.header("Eco Tips Generator")
    keyword = st.text_input("Enter a sustainability keyword (e.g., solar, plastic, transport)")
    if st.button("Get Eco Tips"):
        with st.spinner("Generating tips..."):
            prompt = f"Give me 5 practical eco-friendly tips about: {keyword}"
            tips = call_granite(prompt)
            st.subheader("🌱 Eco Tips")
            st.markdown(tips)

elif menu == "⚡ Anomaly Detection":
    st.header("Energy KPI Anomaly Detection")
    file = st.file_uploader("Upload energy consumption data (CSV)", type="csv")
    if file and st.button("Detect Anomalies"):
        df = pd.read_csv(file)
        if "Zone" in df.columns and "Consumption" in df.columns:
            z_scores = (df["Consumption"] - df["Consumption"].mean()) / df["Consumption"].std()
            df["Anomaly"] = z_scores.abs() > 2
            fig = px.bar(df, x="Zone", y="Consumption", color="Anomaly", title="Anomaly Detection")
            st.subheader("⚠ Detected Anomalies")
            st.dataframe(df)
            st.plotly_chart(fig)
        else:
            st.error("CSV must contain 'Zone' and 'Consumption' columns.")

elif menu == "🤖 Chat Assistant":
    st.header("Ask Your Smart City Assistant")
    user_query = st.text_input("Ask something about your city")
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            prompt = f"Answer the following question about smart cities: {user_query}"
            response = call_granite(prompt)
            st.subheader("Answer")
            st.write(response)