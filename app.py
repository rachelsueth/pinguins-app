
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Preditor de Pinguins", layout="wide")
st.title("🔍 Preditor de Espécie de Pinguim")
st.markdown("Este app usa aprendizado de máquina para prever a **espécie de um pinguim** com base nas medidas corporais.")


st.subheader("📊 Visualização dos Dados")
df = sns.load_dataset("penguins")
df = df.dropna()
st.write(df.head())

fig, ax = plt.subplots()
sns.scatterplot(data=df, x="bill_length_mm", y="body_mass_g", hue="species", ax=ax)
st.pyplot(fig)


st.subheader("🔢 Faça sua própria previsão")
bill_length = st.slider("Comprimento do Bico (mm)", 30.0, 60.0, 45.0)
flipper_length = st.slider("Comprimento da Nadadeira (mm)", 170.0, 230.0, 200.0)
body_mass = st.slider("Peso do Corpo (g)", 2700.0, 6300.0, 4000.0)

if st.button("Prever Espécie"):
    modelo = joblib.load("penguins_model.pkl")
    entrada = [[bill_length, flipper_length, body_mass]]
    predicao = modelo.predict(entrada)[0]
    st.success(f"✅ Espécie prevista: **{predicao}**")
