import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# db connection info
db_user = "postgres"
db_pass = "1234"
db_host = "localhost"
db_port = "5432"
db_name = "marketlens"

# connect
connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

def get_data():
    # just grab the first 100 rows to show it works
    query = "SELECT * FROM raw_ebay_listings LIMIT 100;"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# setup the page
st.set_page_config(page_title="MarketLens Dev", layout="wide")

st.title("MarketLens - Data Feed")
st.text("Week 1: Ingestion Pipeline Check")

try:
    # try to load data from postgres
    df = get_data()
    
    st.success(f"Connected to DB. Showing {len(df)} sample rows.")
    
    # show the table
    st.dataframe(df)
    
except Exception as e:
    st.error(f"Database error: {e}")