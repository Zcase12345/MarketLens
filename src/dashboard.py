import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib
import os
import re

# db config
db_user = "postgres"
db_pass = "1234" 
db_host = "localhost"
db_port = "5432"
db_name = "marketlens"

conn_str = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(conn_str)

# load brain
model_path = os.path.join(os.path.dirname(__file__), 'laptop_price_model.pkl')
try:
    price_model = joblib.load(model_path)
    model_ready = True
except:
    model_ready = False

st.set_page_config(layout="wide")
st.title("MarketLens Pro v0.4")

@st.cache_data
def get_data():
    q = "SELECT brand, model, ram, storage, condition, price_clean FROM raw_ebay_listings WHERE price_clean IS NOT NULL"
    with engine.connect() as c:
        return pd.read_sql(q, c)

df = get_data()

# sidebar
st.sidebar.header("Search Filters")
brands = st.sidebar.multiselect("Brand", df['brand'].unique(), default=df['brand'].unique()[:3])
# helper to extract ram numbers for filter
df['ram_num'] = df['ram'].apply(lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0)

min_ram = st.sidebar.slider("Min RAM", 4, 64, 8)

filtered = df[
    (df['brand'].isin(brands)) & 
    (df['ram_num'] >= min_ram)
]

tab1, tab2, tab3 = st.tabs(["Data View", "Analytics", "Deal Analyzer"])

with tab1:
    st.dataframe(filtered)

with tab2:
    st.subheader("Price by Condition")
    # bar chart showing new vs used prices
    cond_price = filtered.groupby('condition')['price_clean'].mean()
    st.bar_chart(cond_price)

with tab3:
    st.subheader("Smart Deal Analyzer")
    if model_ready:
        c1, c2, c3 = st.columns(3)
        b = c1.selectbox("Brand", df['brand'].unique())
        cond = c2.selectbox("Condition", df['condition'].unique())
        scr = c3.number_input("Screen Size (in)", 11.0, 18.0, 14.0)
        
        c4, c5, c6 = st.columns(3)
        r = c4.number_input("RAM (GB)", 4, 128, 16)
        s = c5.number_input("Storage (GB)", 128, 4000, 512)
        # get listed price
        listed_price = c6.number_input("Listed Price ($)", 0.0, 10000.0, 300.0)
        
        if st.button("Analyze Deal"):
            # make dataframe for model
            input_data = pd.DataFrame({
                'brand': [b],
                'condition': [cond],
                'ram_gb': [r],
                'storage_gb': [s],
                'screen_in': [scr]
            })
            pred = price_model.predict(input_data)[0]
            
            st.metric("AI Market Value", f"${pred:.2f}")
            
            # check for scams
            diff = pred - listed_price
            
            if listed_price < (pred * 0.5):
                st.error(f"HIGH RISK: Price is ${diff:.2f} below market. Likely a scam.")
            elif listed_price < pred:
                st.success(f"GOOD DEAL: Priced ${diff:.2f} below market value.")
            else:
                st.warning(f"BAD DEAL: Priced above fair market value.")
    else:
        st.error("Model missing. Run src/val_model.py")