import streamlit as st
import requests
import plotly.graph_objs as go

st.title("🚀 Interface Trading AI : ML & RL")

if st.button("📊 Entraîner le modèle ML"):
    response = requests.get("http://127.0.0.1:8000/train_ml").json()
    st.success(response["message"])

if st.button("🤖 Choisir une stratégie via ML"):
    response = requests.get("http://127.0.0.1:8000/choose_strategy").json()
    st.success(f"Stratégie recommandée : {response['best_strategy']}")

if st.button("🎯 Entraîner le modèle RL"):
    response = requests.get("http://127.0.0.1:8000/train_rl").json()
    st.success(response["message"])

if st.button("💡 Choisir une stratégie via RL"):
    response = requests.get("http://127.0.0.1:8000/rl_strategy").json()
    st.success(f"Stratégie recommandée par le RL : {response['best_strategy']}")

st.subheader("📈 Visualisation des performances")
backtests = [{"strategy": "sma", "final_value": 10500, "equity_curve": [i * 1.01 for i in range(100)]}]
equity_curve = backtests[0]["equity_curve"]
fig = go.Figure()
fig.add_trace(go.Scatter(y=equity_curve, mode="lines", name="Évolution du portefeuille"))
st.plotly_chart(fig)
