import streamlit as st
import pandas as pd

# Page Configuration (The "Design" Polish starts here)
st.set_page_config(
    page_title="MarketLens",
    page_icon="💸",
    layout="wide"
)

# The Title
st.title("💸 MarketLens")
st.markdown("### The Technical Reseller's Dashboard")

# A Dummy Metric to look impressive immediately
col1, col2, col3 = st.columns(3)
col1.metric("Items Scanned", "1,204", "+12%")
col2.metric("Potential Profit", "$4,250", "+8%")
col3.metric("Top Category", "MacBooks", "High Demand")

# A Sidebar for your "Tools"
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Select Mode:", ["Passive Feed (Aggregator)", "Link Auditor (Validator)"])

st.write(f"Currently viewing: **{mode}**")