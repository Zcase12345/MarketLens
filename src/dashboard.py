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

# try loading model
model_path = os.path.join(os.path.dirname(__file__), 'laptop_price_model.pkl')
try:
    price_model = joblib.load(model_path)
    model_ready = True
except:
    model_ready = False

st.set_page_config(layout="wide")
st.title("MarketLens Pro")

@st.cache_data
def get_data():
    # pull everything
    q = "SELECT brand, model, ram, storage, price_clean FROM raw_ebay_listings WHERE price_clean IS NOT NULL"
    with engine.connect() as c:
        return pd.read_sql(q, c)

df = get_data()

# sidebar filters
st.sidebar.header("Filters")
brands = st.sidebar.multiselect("Brand", df['brand'].unique(), default=df['brand'].unique()[:3])
price_range = st.sidebar.slider("Price", 0, 3000, (0, 3000))
ram_filter = st.sidebar.slider("Min RAM (GB)", 4, 64, 8)
storage_filter = st.sidebar.slider("Min Storage (GB)", 128, 2000, 256)

# filter logic
filtered = df[
    (df['brand'].isin(brands)) & 
    (df['price_clean'] >= price_range[0]) & 
    (df['price_clean'] <= price_range[1]) &
    (df['ram'].apply(lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0) >= ram_filter) &
    (df['storage'].apply(lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0) >= storage_filter)
]

# main tabs
tab1, tab2, tab3 = st.tabs(["Data", "Charts", "Predictor"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(filtered)
    st.caption(f"Rows: {len(filtered)}")

with tab2:
    st.subheader("Price Stats")
    
    # avg price bar chart
    avg_prices = filtered.groupby('brand')['price_clean'].mean().sort_values()
    st.bar_chart(avg_prices)
    
    # market share pie chart
    st.subheader("Volume Share")
    fig, ax = plt.subplots()
    filtered['brand'].value_counts().head(5).plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Value Estimator")
    
    if model_ready:
        c1, c2, c3 = st.columns(3)
        b = c1.selectbox("Brand", df['brand'].unique())
        r = c2.number_input("RAM (GB)", 4, 128, 16)
        s = c3.number_input("Storage (GB)", 128, 4000, 512)
        
        if st.button("Get Price"):
            # format input for model
            input_df = pd.DataFrame({'brand': [b], 'ram_gb': [r], 'storage_gb': [s]})
            pred = price_model.predict(input_df)[0]
            st.metric("Estimated Value", f"${pred:.2f}")
    else:
        st.warning("Run val_model.py first!")