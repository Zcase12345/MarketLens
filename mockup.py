import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# --- PART 1: SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="MarketLens")

# Force Dark Mode & Hide Default Menu
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- PART 2: SIDEBAR NAVIGATION ---
st.sidebar.title("MarketLens")
st.sidebar.write("User: **Technical Reseller**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", ["Aggregator Feed", "Link Auditor", "Analytics"])

st.sidebar.markdown("---")
st.sidebar.subheader("Active Filters")
# Fixed Source to Craigslist as requested
st.sidebar.info("Source: **Craigslist**")
category = st.sidebar.selectbox("Category", ["Laptops", "Desktops", "Components"], index=0)
min_profit = st.sidebar.slider("Min Profit Margin", 0, 500, 150)

if st.sidebar.button("Refresh Feed"):
    st.sidebar.success("Feed Updated!")

# --- PART 3: AGGREGATOR FEED PAGE ---
if page == "Aggregator Feed":
    st.title("MarketLens Dashboard")
    st.caption("Live Arbitrage Intelligence System • v1.0.0")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Profit Opportunity", "$3,150", "+8%")
    col2.metric("Verified Deals", "12", "+2")
    col3.metric("Avg. ROI", "28%", "+1.2%")
    col4.metric("Scanned Today", "842", "Active")

    st.markdown("---")

    # Fake Data for Chart (Laptop Prices Trend)
    fig = go.Figure(data=go.Scatter(
        x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        y=[2800, 3100, 2900, 3250, 3600, 3900, 3150],
        mode='lines+markers',
        name='Profit Potential',
        line=dict(color='#00CC96', width=3)
    ))
    fig.update_layout(
        title="Weekly Profit Opportunity (Laptops Only)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Fake Data for Table - ALL LAPTOPS, ALL CRAIGSLIST
    data = {
        "Item Name": [
            "MacBook Pro M1 14 (16GB/512GB)", 
            "Dell XPS 13 9310 (i7/16GB)", 
            "Lenovo ThinkPad X1 Carbon Gen 9", 
            "HP Spectre x360 14-inch", 
            "MacBook Air M2 (Midnight)"
        ],
        "Listing Price": [850, 600, 750, 500, 900],
        "Market Value": [1200, 850, 1050, 800, 1050],
        "Source": ["Craigslist", "Craigslist", "Craigslist", "Craigslist", "Craigslist"],
        "Condition": ["Good", "Like New", "Excellent", "Fair", "Open Box"]
    }
    df = pd.DataFrame(data)
    df["Projected Profit"] = df["Market Value"] - df["Listing Price"]
    df["ROI"] = ((df["Projected Profit"] / df["Listing Price"]) * 100).round(1).astype(str) + "%"
    
    # Filter Logic Simulation
    df = df[df["Projected Profit"] > 0]

    st.subheader("Live Opportunities: Laptops")
    st.dataframe(
        df,
        column_config={
            "Listing Price": st.column_config.NumberColumn(format="$%d"),
            "Market Value": st.column_config.NumberColumn(format="$%d"),
            "Projected Profit": st.column_config.NumberColumn(format="$%d"),
        },
        use_container_width=True,
        hide_index=True
    )

    # Action Simulation
    selected_item = st.selectbox("Select Item to Audit:", df["Item Name"])
    if st.button("Analyze Selected Item"):
        st.info(f"Redirecting {selected_item} to Deep Analysis...")

# --- PART 4: LINK AUDITOR PAGE ---
elif page == "Link Auditor":
    st.title("Active Link Auditor")
    st.info("Paste a Craigslist URL to run the Valuation Engine.")

    col_input, col_btn = st.columns([4, 1])
    # Updated placeholder to match the Craigslist theme
    url = col_input.text_input("Paste URL", placeholder="https://stlouis.craigslist.org/sys/...")
    analyze = col_btn.button("Run Audit", type="primary")

    if analyze:
        # Simulate Processing Time (The "Magic" Effect)
        with st.spinner('Scraping Craigslist listing...'):
            time.sleep(1)
        with st.spinner('Running Computer Vision (CLIP) model on images...'):
            time.sleep(1.5)
        with st.spinner('Calculating Fair Market Value (Scikit-Learn)...'):
            time.sleep(1)
        
        st.success("Analysis Complete! Deal Detected.")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        # Left Column: Image & Vision
        with res_col1:
            st.image("https://placehold.co/400x300/262730/white?text=Laptop+Image", caption="Listing Image Source")
            st.caption("CLIP Verified: Laptop/Clamshell Device (99%)")
        
        # Right Column: Data & Valuation
        with res_col2:
            st.header("MacBook Pro 14-inch (M1 Pro)")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Listing Price", "$850")
            m2.metric("AI Valuation", "$1,200")
            m3.metric("Profit Spread", "$350", "+41%")
            
            st.markdown("### Intelligence Report")
            st.write("**Pros:** Price is significantly below market. 'OBO' in description suggests negotiation room.")
            st.write("**Cons:** Seller account is new (Risk Factor).")
            
            st.write("Risk Score:")
            st.progress(35)
            st.caption("Medium-Low Risk (35/100)")

# --- PART 5: ANALYTICS PAGE ---
elif page == "Analytics":
    st.title("Market Trends")
    st.write("Historical price data for tracked laptops.")
    
    # Reuse chart for visual filler
    fig = go.Figure(data=go.Scatter(
        x=["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        y=[950, 920, 880, 850, 800, 780],
        mode='lines',
        name='MacBook Air M1 Avg Price',
        line=dict(color='#FF4B4B', width=3)
    ))
    fig.update_layout(title="MacBook Air M1 Price Depreciation (6 Months)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)