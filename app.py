import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Supply Chain Analytics", layout="wide")

# --------------------------------------------------
# PROFESSIONAL YELLOW UI
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #F4C430 0%, #F7D046 100%);
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #1C1C1C;
    font-size: 40px;
    font-weight: 700;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.metric-title {
    color: #666;
    font-size: 14px;
}
.metric-value {
    color: #111;
    font-size: 28px;
    font-weight: 700;
}
.stButton>button {
    background-color: #111;
    color: white;
    border-radius: 8px;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Analytics Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')

data = load_data()

# --------------------------------------------------
# METRICS
# --------------------------------------------------

total = len(data)
delayed = data['Late_delivery_risk'].sum()
on_time = total - delayed
delay_rate = (delayed / total) * 100

def metric_card(title, value):
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Total Shipments", total)
with col2:
    metric_card("Delayed Shipments", delayed)
with col3:
    metric_card("On-Time Shipments", on_time)
with col4:
    metric_card("Delay Rate (%)", f"{delay_rate:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# VISUALS
# --------------------------------------------------

colA, colB = st.columns(2)

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig_pie = px.pie(
        names=["On-Time", "Delayed"],
        values=[on_time, delayed],
        color_discrete_sequence=["#2ECC71", "#E74C3C"]
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        data,
        x="Sales",
        y="Benefit per order",
        opacity=0.4
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("model.pkl")

st.subheader("Shipment Delay Prediction")

st.markdown('<div class="card">', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    scheduled = st.number_input("Scheduled Days", 1, 10, 3)
    quantity = st.number_input("Quantity", 1, 10, 2)

with c2:
    price = st.number_input("Product Price", 1.0, 10000.0, 100.0)
    sales_input = st.number_input("Sales", 1.0, 20000.0, 200.0)

with c3:
    benefit = st.number_input("Profit", -1000.0, 10000.0, 50.0)
    discount = st.slider("Discount Rate", 0.0, 1.0, 0.1)

if st.button("Predict Shipment Status"):

    input_data = np.array([[scheduled, quantity, price, benefit, sales_input, discount]])

    proba = model.predict_proba(input_data)[0]
    delay_prob = proba[1]
    ontime_prob = proba[0]

    if delay_prob > 0.5:
        st.error(f"Shipment likely delayed ({delay_prob*100:.1f}% confidence)")
    else:
        st.success(f"Shipment likely on-time ({ontime_prob*100:.1f}% confidence)")

st.markdown('</div>', unsafe_allow_html=True)