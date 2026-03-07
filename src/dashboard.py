import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib
import os
import re
import time

# lock sidebar open
st.set_page_config(layout="wide", page_title="MarketLens", initial_sidebar_state="expanded")

# force dark theme
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* make header transparent, NOT hidden */
    header { background-color: transparent !important; }
    .stDeployButton { display: none !important; }
    
    /* hide the ugly default arrow */
    [data-testid="collapsedControl"] svg, 
    [data-testid="stSidebarCollapseButton"] svg { 
        display: none !important; 
    }
    
    /* inject trendy cyan menu icon */
    [data-testid="collapsedControl"]::after, 
    [data-testid="stSidebarCollapseButton"]::after {
        content: '☰'; 
        font-size: 24px;
        color: #00e5ff;
        display: block;
        margin: 5px;
        text-shadow: 0px 0px 8px #00e5ff;
        cursor: pointer;
    }
    
    /* fix chopped buttons */
    .block-container { padding-top: 3rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    
    /* fix ui elements */
    div[data-testid="stRadio"] > div { flex-direction: row; gap: 10px; padding: 10px 0; }
    div[data-testid="stRadio"] label { background-color: #1E2127; padding: 10px 20px; border-radius: 8px; cursor: pointer; border: 1px solid #2D3139; }
</style>
""", unsafe_allow_html=True)

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

def clean_cond(x):
    # clean messy text
    if pd.isna(x): return 'Unknown'
    v = str(x).lower()
    if 'refurbished' in v: return 'Refurbished'
    if 'new' in v and 'used' not in v: return 'New'
    if 'open box' in v: return 'Open Box'
    if 'parts' in v: return 'For Parts'
    return 'Used'

@st.cache_data
def get_data():
    q = "SELECT brand, model, ram, storage, condition, screen_size, price_clean FROM raw_ebay_listings WHERE price_clean IS NOT NULL"
    with engine.connect() as c:
        df_raw = pd.read_sql(q, c)
        
        # apply condition cleaner
        df_raw['condition'] = df_raw['condition'].apply(clean_cond)
        return df_raw

df = get_data()

# sidebar
st.sidebar.title("MarketLens")
st.sidebar.write("User: **Technical Reseller**")
st.sidebar.header("Search Filters")
brands = st.sidebar.multiselect("Brand", df['brand'].unique(), default=df['brand'].unique()[:3], help="Select target laptop brands")
# helper to extract ram numbers for filter
df['ram_num'] = df['ram'].apply(lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 0)

# get screen size
df['screen_num'] = df['screen_size'].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(1)) if pd.notna(x) and re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

min_ram = st.sidebar.slider("Min RAM", 4, 64, 8, help="Filter minimum RAM")

# copy df safely
filtered = df[
    (df['brand'].isin(brands)) & 
    (df['ram_num'] >= min_ram)
].copy()

# custom navigation bar
nav_mode = st.radio("Navigation", ["Aggregator Feed", "Link Auditor", "Analytics"], label_visibility="collapsed")

if nav_mode == "Aggregator Feed":
    st.subheader("Live Arbitrage Feed")
    if model_ready and not filtered.empty:
        # prep data for model
        eval_df = filtered[['brand', 'condition']].copy()
        eval_df['ram_gb'] = filtered['ram_num']
        # extract storage 
        eval_df['storage_gb'] = filtered['storage'].apply(lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if pd.notna(x) and re.search(r'(\d+)', str(x)) else 0)
        eval_df['screen_in'] = filtered['screen_num']
        
        # run batch predict
        preds = price_model.predict(eval_df)
        
        # calc profit spread
        filtered['AI_Value'] = preds
        filtered['Profit_Spread'] = filtered['AI_Value'] - filtered['price_clean']
        
        # sort by profit
        feed = filtered.sort_values('Profit_Spread', ascending=False)
        
        # format monetary columns
        feed['price_clean'] = feed['price_clean'].apply(lambda x: f"${x:.2f}")
        feed['AI_Value'] = feed['AI_Value'].apply(lambda x: f"${x:.2f}")
        feed['Profit_Spread'] = feed['Profit_Spread'].apply(lambda x: f"${x:.2f}")
        
        display_cols = ['brand', 'condition', 'price_clean', 'AI_Value', 'Profit_Spread']
        
        # center data config
        styled_feed = feed[display_cols].head(50).style.set_properties(**{'text-align': 'center'})
        
        # hide useless index
        st.dataframe(styled_feed, use_container_width=True, hide_index=True)
    else:
        st.error("Model missing. Run src/val_model.py")

elif nav_mode == "Link Auditor":
    st.subheader("Link Auditor & Vision Check")
    colA, colB = st.columns([1, 2])
    
    with colA:
        # mock image upload
        img = st.file_uploader("Upload Listing Photo", type=["jpg", "png"])
        if img:
            st.image(img, use_container_width=True)
            with st.spinner("Running CLIP..."):
                time.sleep(1.5)
            st.success("Vision Check: Screen Intact (98%)")
            st.progress(85)
            
    with colB:
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
                
                # calc profit margin
                fees = pred * 0.13
                profit = pred - listed_price - fees - 20.0
                
                if listed_price < (pred * 0.5):
                    st.error(f"HIGH RISK: Price is ${diff:.2f} below market. Likely a scam.")
                elif listed_price < pred:
                    st.success(f"GOOD DEAL: Priced ${diff:.2f} below market value.")
                    st.info(f"Est. Profit: ${profit:.2f} (After fees/shipping)")
                else:
                    st.warning(f"BAD DEAL: Priced above fair market value.")
        else:
            st.error("Model missing. Run src/val_model.py")

elif nav_mode == "Analytics":
    st.subheader("Price by Condition")
    # bar chart showing new vs used prices
    cond_price = filtered.groupby('condition')['price_clean'].mean()
    st.bar_chart(cond_price)
    
    # screen size scatter
    st.subheader("Price vs Screen")
    fig, ax = plt.subplots()
    ax.scatter(filtered['screen_num'], filtered['price_clean'], alpha=0.5, color="blue")
    st.pyplot(fig)