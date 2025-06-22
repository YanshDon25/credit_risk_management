
import streamlit as st
import pandas as pd
from preprocess_and_model import model

st.set_page_config(page_title="Credit Risk Model", layout="centered")
st.title("📊 Credit Risk Scoring (Alt-Data)")

st.markdown("Enter alternative behavior data to assess probability of default (PD).")

# Input form
recharge = st.slider("Monthly Mobile Recharge (INR)", 50, 500, 200)
drop_rate = st.slider("Call Drop Rate", 0.0, 0.3, 0.1)
social_media = st.slider("Daily Social Media Usage (hours)", 0.0, 6.0, 3.0)
electricity = st.selectbox("Electricity Bill Paid On Time?", [1, 0])
gas = st.selectbox("Gas Bill Paid On Time?", [1, 0])
apps = st.slider("App Installs (Last 30 Days)", 0, 15, 4)
phone = st.selectbox("Smartphone Type", ["low_end", "mid_range", "premium"])
contacts = st.slider("Emergency Contacts Count", 0, 10, 2)
geo_change = st.slider("Geo Location Change Rate", 0.0, 0.5, 0.2)

# Construct dataframe
input_df = pd.DataFrame([{
    'monthly_mobile_recharge': recharge,
    'call_drop_rate': drop_rate,
    'social_media_usage_hrs': social_media,
    'electricity_bill_paid_on_time': electricity,
    'gas_bill_paid_on_time': gas,
    'app_installs_last_30_days': apps,
    'smartphone_type': phone,
    'emergency_contacts_count': contacts,
    'geo_location_change_rate': geo_change
}])

# Predict
if st.button("Predict Credit Risk"):
    score = model.predict_proba(input_df)[0][1]
    risk_band = "Low" if score < 0.3 else "Medium" if score < 0.6 else "High"

    st.subheader(f"📉 Probability of Default (PD): `{score:.2f}`")
    st.markdown(f"### 🔐 **Risk Category: `{risk_band}`**")
