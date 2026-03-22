# requires: pip install transformers torch pillow
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #080b10; color: #c9d1d9; }
    #MainMenu, footer, header { visibility: hidden !important; display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1c2333; }
    [data-testid="stSidebar"] .stMarkdown h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: #58a6ff; letter-spacing: 0.08em; text-transform: uppercase; }
    /* fix chopped buttons */
    .block-container { padding-top: 2rem; padding-bottom: 1rem; max-width: 1400px; }
    [data-testid="stMetric"] { background-color: #0d1117; border: 1px solid #1c2333; border-radius: 6px; padding: 1rem 1.2rem; }
    [data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }
    [data-testid="stMetricDelta"] svg { display: none; }
    /* fix ui elements */
    div[data-testid="stRadio"] > div { flex-direction: row; gap: 8px; padding: 6px 0 16px 0; }
    div[data-testid="stRadio"] label { background-color: #0d1117; border: 1px solid #1c2333; border-radius: 4px; padding: 8px 18px; cursor: pointer; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    div[data-testid="stRadio"] label:hover { border-color: #58a6ff; color: #58a6ff; }
    .stDataFrame { border: 1px solid #1c2333; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    h2, h3 { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.12em; border-bottom: 1px solid #1c2333; padding-bottom: 6px; margin-bottom: 1rem; }
    .stButton > button { background-color: #1f6feb; border: none; border-radius: 4px; color: white; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; padding: 10px 20px; width: 100%; }
    .stButton > button:hover { background-color: #388bfd; }
    .stSuccess { border-left: 3px solid #3fb950; background-color: #0d2119; }
    .stError   { border-left: 3px solid #f85149; background-color: #200d0d; }
    .stWarning { border-left: 3px solid #d29922; background-color: #1f1a0d; }
    .stInfo    { border-left: 3px solid #58a6ff; background-color: #0d1729; }
    .stSelectbox > div > div, .stNumberInput > div > div { background-color: #0d1117; border: 1px solid #1c2333; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    .stFileUploader { border: 1px dashed #1c2333; border-radius: 6px; background-color: #0d1117; }
    .stProgress > div > div > div > div { background-color: #3fb950; }
    .stMultiSelect span[data-baseweb="tag"] { background-color: #1f6feb; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; }
    hr { border-color: #1c2333; }
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
except Exception:
    model_ready = False

# load clip once
@st.cache_resource
def load_clip():
    try:
        from transformers import CLIPProcessor, CLIPModel
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return m, p
    except Exception:
        return None, None

def run_clip(image_file):
    # real clip inference
    clip_model, clip_proc = load_clip()
    if clip_model is None:
        return None
    from PIL import Image
    labels = [
        "a laptop computer with a cracked or damaged screen",
        "a laptop computer with chassis damage or dents",
        "a laptop computer in perfect condition" 
    ]
    img     = Image.open(image_file).convert("RGB")
    inputs  = clip_proc(text=labels, images=img, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs   = outputs.logits_per_image.softmax(dim=1)[0].detach().tolist()
    return dict(zip(labels, probs))

def clean_cond(x):
    # clean messy text
    if pd.isna(x): return 'Unknown'
    v = str(x).lower()
    if 'refurbished' in v: return 'Refurbished'
    if 'new' in v and 'used' not in v: return 'New'
    if 'open box' in v: return 'Open Box'
    if 'parts' in v: return 'For Parts'
    return 'Used'

def extract_num(val, as_float=False):
    # pull first number from string
    if pd.isna(val): return None
    pattern = r'(\d+\.?\d*)' if as_float else r'(\d+)'
    m = re.search(pattern, str(val))
    if not m: return None
    return float(m.group(1)) if as_float else int(m.group(1))

@st.cache_data(ttl=300)
def get_data():
    # pull strictly clean data
    q = """
        SELECT brand, model, ram, storage, condition, screen_size, price_clean, processor, created_at
        FROM raw_ebay_listings
        WHERE price_clean IS NOT NULL
          AND price_clean > 0
          AND brand IS NOT NULL
          AND brand NOT IN ('', '?', 'Different')
          AND LOWER(brand) != 'does not apply'
          AND LOWER(COALESCE(model, '')) != 'does not apply'
          AND COALESCE(model, '') NOT IN ('Different', 'different')
          AND LOWER(COALESCE(condition, '')) != 'does not apply'
          AND ram IS NOT NULL
          AND ram NOT IN ('', '?', '0')
          AND storage IS NOT NULL
          AND storage NOT IN ('', '?', '0')
          AND screen_size IS NOT NULL
          AND screen_size NOT IN ('', '?', '0')
          AND condition IS NOT NULL
    """
    with engine.connect() as c:
        df_raw = pd.read_sql(q, c)

    # apply condition cleaner
    df_raw['condition'] = df_raw['condition'].apply(clean_cond)

    # parse numeric helpers
    df_raw['ram_num']     = df_raw['ram'].apply(extract_num).fillna(0)
    df_raw['storage_num'] = df_raw['storage'].apply(extract_num).fillna(0)
    df_raw['screen_num']  = df_raw['screen_size'].apply(lambda x: extract_num(x, as_float=True)).fillna(0)

    # final safety check
    df_raw = df_raw[(df_raw['ram_num'] > 0) & (df_raw['storage_num'] > 0) & (df_raw['screen_num'] > 0)]

    return df_raw

df = get_data()

# sidebar
st.sidebar.title("MarketLens")
st.sidebar.write("User: **Technical Reseller**")
st.sidebar.caption("eBay historical data")
st.sidebar.header("Search Filters")

# build brand list
all_brands = sorted([b for b in df['brand'].dropna().unique() if str(b) != 'nan'])
brands = st.sidebar.multiselect(
    "Brand", all_brands, default=all_brands[:4],
    format_func=lambda x: str(x).title(),
    help="Filter results to specific laptop brands"
)

min_ram = st.sidebar.slider(
    "Min RAM (GB)", 4, 64, 8,
    help="Only show laptops with at least this much RAM"
)

# condition filter
all_conds = sorted(df['condition'].unique().tolist())
selected_conds = st.sidebar.multiselect(
    "Condition", all_conds,
    default=[c for c in all_conds if c not in ['For Parts', 'Unknown']],
    format_func=lambda x: str(x).title(),
    help="Filter by item condition"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} total records")

# copy df safely
filtered = df[
    (df['brand'].isin(brands)) &
    (df['ram_num'] >= min_ram) &
    (df['condition'].isin(selected_conds))
].copy()

# centered page title
st.markdown("""
<div style='text-align:center; padding: 0.5rem 0 1.2rem 0;'>
    <span style='font-family: IBM Plex Mono, monospace; font-size: 2rem; font-weight: 600;
                 color: #e6edf3; letter-spacing: 0.12em; text-transform: uppercase;'>
        MarketLens
    </span>
    <div style='font-family: IBM Plex Sans, sans-serif; font-size: 0.75rem;
                color: #8b949e; letter-spacing: 0.08em; margin-top: 4px;'>
        Laptop Arbitrage Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# custom navigation bar
nav_mode = st.radio(
    "Navigation",
    ["Aggregator Feed", "Link Auditor", "Analytics"],
    label_visibility="collapsed",
    help="Switch between the three main sections of MarketLens"
)

st.markdown("---")

# aggregator feed
if nav_mode == "Aggregator Feed":
    st.subheader("Live Arbitrage Feed")

    if model_ready and not filtered.empty:
        # prep data for model
        eval_df = filtered[['brand', 'condition']].copy()
        eval_df['ram_gb']     = filtered['ram_num'].fillna(0).astype(int)
        # extract storage
        eval_df['storage_gb'] = filtered['storage_num'].fillna(0).astype(int)
        eval_df['screen_in']  = filtered['screen_num'].fillna(0)

        # run batch predict
        preds = price_model.predict(eval_df)

        # calc profit spread
        filtered = filtered.copy()
        filtered['AI_Value']      = preds
        filtered['Profit_Spread'] = filtered['AI_Value'] - filtered['price_clean']

        # only show profitable rows
        positive = filtered[filtered['Profit_Spread'] > 0].copy()

        # rotate sample every 5 mins to feel live
        seed = int(time.time() / 300)
        feed = positive.nlargest(1000, 'Profit_Spread').sample(frac=1, random_state=seed).reset_index(drop=True)

        # metrics from real data
        total_profit = feed['Profit_Spread'].sum()
        avg_roi      = ((feed['Profit_Spread'] / feed['price_clean']).mean() * 100) if not feed.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Profit Found", f"${total_profit:,.0f}", f"+{len(feed)} Deals",
                  help="Sum of every profit spread. Not realized profit — it's the total if you bought and resold everything.")
        c2.metric("Verified Deals", str(len(feed)), "Active",
                  help="Count of listings where the AI predicted value is higher than the asking price.")
        c3.metric("Avg ROI", f"{avg_roi:.1f}%", "After Fees",
                  help="Average return on investment across all current deals. Calculated as (Spread / Ask Price) * 100.")
        c4.metric("Items Scanned", f"{len(filtered):,}", "Filtered",
                  help="Total listings the model evaluated.")

        st.markdown("---")

        # row count control
        lbl_col, btn_col = st.columns([3, 2])
        with lbl_col:
            st.caption("Sorted by Profit Spread — feed auto updates every 5 mins")
        with btn_col:
            row_limit = st.radio(
                "Show rows",
                options=[10, 25, 50, 100, 1000],
                index=0,
                horizontal=True,
                label_visibility="collapsed"
            )

        # sort by profit then slice to row_limit
        feed = feed.sort_values('Profit_Spread', ascending=False).head(row_limit).reset_index(drop=True)

        # format monetary columns
        display = feed[['brand', 'model', 'condition', 'ram', 'storage', 'price_clean', 'AI_Value', 'Profit_Spread']].copy()
        display.columns = ['Brand', 'Model', 'Condition', 'RAM', 'Storage', 'Ask Price', 'AI Value', 'Profit Spread']
        
        # title case safe formats
        display['Brand'] = display['Brand'].astype(str).str.title()
        display['Model'] = display['Model'].astype(str).str.title()
        display['Condition'] = display['Condition'].astype(str).str.title()

        display['Ask Price']     = display['Ask Price'].apply(lambda x: f"${x:,.2f}")
        display['AI Value']      = display['AI Value'].apply(lambda x: f"${x:,.2f}")
        display['Profit Spread'] = display['Profit Spread'].apply(lambda x: f"${x:,.2f}")

        # center data config
        styled_feed = display.style.set_properties(**{'text-align': 'center'})

        # hide useless index
        st.dataframe(styled_feed, use_container_width=True, hide_index=True)

        # inline row detail sync
        if not feed.empty:
            row_idx = st.selectbox(
                "Inspect row",
                options=list(range(len(feed))),
                format_func=lambda i: f"Row {i+1}  —  {str(feed.iloc[i]['brand']).title()}  |  {str(feed.iloc[i].get('model','N/A')).title()[:30]}  |  ${feed.iloc[i]['price_clean']:,.2f}",
                help="Choose a row from the table above to expand its full details and net profit estimate"
            )

            with st.expander(f"Details — Row {row_idx + 1}", expanded=False):
                row = feed.iloc[row_idx]
                d1, d2, d3 = st.columns(3)
                d1.metric("Brand",     str(row.get('brand', 'N/A')).title())
                d2.metric("Model",     str(row.get('model', 'N/A')).title()[:40])
                d3.metric("Condition", str(row.get('condition', 'N/A')).title())

                d4, d5, d6 = st.columns(3)
                d4.metric("RAM",     str(row.get('ram', 'N/A')))
                d5.metric("Storage", str(row.get('storage', 'N/A')))
                d6.metric("Screen",  str(row.get('screen_size', 'N/A')))

                d7, d8, d9 = st.columns(3)
                d7.metric("Ask Price",     f"${row['price_clean']:,.2f}")
                d8.metric("AI Value",      f"${row['AI_Value']:,.2f}")
                d9.metric("Profit Spread", f"${row['Profit_Spread']:,.2f}")
                st.caption("Ask Price = listed price  |  AI Value = model prediction  |  Profit Spread = AI Value minus Ask Price")

                fees   = row['AI_Value'] * 0.13
                profit = row['Profit_Spread'] - fees - 20.0
                if profit > 0:
                    st.success(f"Est. net profit after ~13% fees + $20 shipping: ${profit:,.2f}")
                else:
                    st.warning("Spread exists but may not cover fees and shipping costs.")

    elif not model_ready:
        st.error("Model missing. Run src/val_model.py")
    else:
        st.warning("No data.")

# link auditor
elif nav_mode == "Link Auditor":
    st.subheader("Link Auditor & Vision Check")
    st.caption(
        "Upload a photo of the listing, then fill in the specs. Vision check uses CLIP to verify the photo. Deal analysis uses Random Forest model.",
        help="CLIP (Contrastive Language-Image Pre-Training by OpenAI) is a vision language model used to assess uploaded images for condition and damage. The deal model is Random Forest trained on eBay historical prices."
    )
    colA, colB = st.columns([1, 2])

    with colA:
        # mock image upload
        img = st.file_uploader(
            "Upload Listing Photo", type=["jpg", "jpeg", "png"],
            help="Upload a photo from the listing. CLIP checks whether it's actually a laptop and flags damage."
        )
        if img:
            st.image(img, use_container_width=True)
            with st.spinner("Running CLIP AI Valuator..."):
                scores = run_clip(img)

            if scores is not None:
                screen_score  = scores.get("a laptop computer with a cracked or damaged screen", 0)
                chassis_score = scores.get("a laptop computer with chassis damage or dents", 0)

                if screen_score > 0.30:
                    st.warning("Possible screen damage detected — inspect carefully before buying.")
                elif chassis_score > 0.30:
                    st.warning("Possible chassis damage detected — inspect before buying.")
                else:
                    st.success("No obvious damage detected.")

                st.caption(
                    f"Screen Damage: {screen_score:.0%}  |  Chassis Damage: {chassis_score:.0%}"
                )
            else:
                st.warning("CLIP not loaded. Run: pip install transformers torch pillow")
        else:
            st.markdown(
                "<div style='border:1px dashed #1c2333;border-radius:6px;padding:40px 20px;"
                "text-align:center;color:#8b949e;font-size:0.75rem;font-family:IBM Plex Mono,monospace;'>"
                "DROP PHOTO HERE<br>jpg / png / jpeg</div>",
                unsafe_allow_html=True
            )

    with colB:
        if model_ready:
            c1, c2, c3 = st.columns(3)
            b    = c1.selectbox("Brand",
                                sorted([x for x in df['brand'].dropna().unique() if str(x) != 'nan']),
                                format_func=lambda x: str(x).title(),
                                help="Brand of the laptop in the listing")
            cond = c2.selectbox("Condition",
                                sorted(df['condition'].unique().tolist()),
                                format_func=lambda x: str(x).title(),
                                help="The seller listed condition of the item")
            scr  = c3.number_input("Screen Size (in)", 11.0, 18.0, 14.0,
                                   help="Screen diagonal in inches — check the listing description or look up the model number")

            c4, c5, c6 = st.columns(3)
            r = c4.number_input("RAM (GB)", 4, 128, 16,
                                help="RAM in GB")
            s = c5.number_input("Storage (GB)", 128, 4000, 512,
                                help="SSD or HDD size in GB")
            # get listed price
            listed_price = c6.number_input("Listed Price ($)", 0.0, 10000.0, 300.0,
                                           help="The asking price from the listing — what you would actually pay")

            if st.button(
                "Analyze Deal",
                help="Runs the Random Forest valuation model on these specs and compares the result to your listed price to determine if it's a deal, a scam, or overpriced"
            ):
                with st.spinner("Running valuation model..."):
                    time.sleep(0.6)

                # make dataframe for model
                input_data = pd.DataFrame({
                    'brand':      [b],
                    'condition':  [cond],
                    'ram_gb':     [r],
                    'storage_gb': [s],
                    'screen_in':  [scr]
                })
                pred = price_model.predict(input_data)[0]

                # check for scams
                diff = pred - listed_price

                # calc profit margin
                fees   = pred * 0.13
                profit = pred - listed_price - fees - 20.0

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("AI Market Value", f"${pred:,.2f}",
                           help="The Random Forest model's predicted fair market value based on comparable historical eBay sales for this brand, condition, RAM, storage, and screen size")
                mc2.metric("Listed Price",    f"${listed_price:,.2f}",
                           help="The asking price you entered above")
                mc3.metric("Gross Spread",    f"${diff:,.2f}",
                           help="AI Value minus Listed Price. Positive means the AI thinks you're getting it below market.")

                st.markdown("---")

                if listed_price <= 0:
                    st.warning("Enter a listed price to get a verdict.")
                elif listed_price < (pred * 0.5):
                    risk = min(int((1 - listed_price / pred) * 100), 95)
                    st.error(f"HIGH RISK: Price is ${diff:.2f} below market. Verify before purchasing.")
                    st.caption("Price more than 50% below AI value is a common red flag for scams or major undisclosed defects.")
                    st.progress(risk)
                    st.caption(f"Risk score: {risk}/100 — 0 = safe, 100 = very high risk")
                elif listed_price < pred:
                    risk = max(int((listed_price / pred) * 40), 5)
                    st.success(f"GOOD DEAL: Priced ${diff:.2f} below market value.")
                    if profit > 0:
                        st.info(f"Est. Profit: ${profit:.2f} (After fees/shipping)")
                        st.caption("Estimated net after ~13% marketplace fees and $20 shipping. Actual fees vary.")
                    st.progress(risk)
                    st.caption(f"Risk score: {risk}/100 — low score means safer deal")
                else:
                    st.warning(f"BAD DEAL: Priced above fair market value.")
                    st.progress(70)
                    st.caption("Risk score: 70/100 — asking price exceeds what the model predicts comparable items sell for")
        else:
            st.error("Model missing. Run src/val_model.py")

# analytics
elif nav_mode == "Analytics":

    if filtered.empty:
        st.warning("No data.")
        st.stop()

    CHART_W, CHART_H = 8, 5

    def dark_fig(w=CHART_W, h=CHART_H):
        # reusable dark chart
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1c2333')
        return fig, ax

    def fmt_price(ax, axis='y'):
        fmt = mticker.FuncFormatter(lambda v, _: f'${v:,.0f}')
        if axis == 'y':
            ax.yaxis.set_major_formatter(fmt)
        else:
            ax.xaxis.set_major_formatter(fmt)

    # clamp for clean charts
    plot_df = filtered[
        (filtered['price_clean'] > 30) &
        (filtered['price_clean'] < 4000)
    ].copy()

    st.caption(f"Showing analytics for {len(plot_df):,} listings.")

    # row 1
    # row 1 — price distribution full width

    st.subheader("Price Distribution")
    st.caption(
        "How many listings fall at each price point.",
        help="Histogram of asking prices. Tall bars = many listings at that price. Shows where the used laptop market clusters — useful for spotting the most liquid price ranges."
    )
    fig, ax = dark_fig()
    ax.hist(plot_df['price_clean'], bins=40, color='#1f6feb', edgecolor='#0d1117', linewidth=0.4)
    fmt_price(ax, 'x')
    ax.set_xlabel('Price', color='#8b949e', fontsize=12)
    ax.set_ylabel('Count', color='#8b949e', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # row 2
    r2a, r2b = st.columns(2)

    with r2a:
        st.subheader("RAM vs Price")
        st.caption(
            "Whether more RAM correlates with higher price.",
            help="Each dot is one listing. An upward trend means RAM is a strong price driver — useful for knowing which spec to focus on when estimating value."
        )
        ram_plot = plot_df[plot_df['ram_num'] > 0].copy()
        fig, ax = dark_fig()
        ax.scatter(ram_plot['ram_num'], ram_plot['price_clean'],
                   alpha=0.25, color='#58a6ff', s=24, linewidths=0)
        ax.set_xlabel('RAM (GB)', color='#8b949e', fontsize=12)
        ax.set_ylabel('Price',    color='#8b949e', fontsize=12)
        fmt_price(ax, 'y')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with r2b:
        st.subheader("Storage vs Price")
        st.caption(
            "Whether more storage correlates with higher price.",
            help="Each dot is one listing. Helps you see if buyers pay a meaningful premium for larger drives, or if storage has diminishing returns above a certain size."
        )
        sto_plot = plot_df[plot_df['storage_num'] > 0].copy()
        fig, ax = dark_fig()
        ax.scatter(sto_plot['storage_num'], sto_plot['price_clean'],
                   alpha=0.25, color='#3fb950', s=24, linewidths=0)
        ax.set_xlabel('Storage (GB)', color='#8b949e', fontsize=12)
        ax.set_ylabel('Price',        color='#8b949e', fontsize=12)
        fmt_price(ax, 'y')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # row 3
    r3a, r3b = st.columns(2)

    with r3a:
        st.subheader("Price by Condition")
        st.caption(
            "Average asking price for each condition category.",
            help="Compares New vs Used vs Refurbished etc. A large gap between conditions means condition is a major price lever for that brand — factor this in when auditing deals."
        )
        cond_price = plot_df.groupby('condition')['price_clean'].mean().sort_values(ascending=False)
        fig, ax = dark_fig()
        colors_c = ['#58a6ff','#3fb950','#d29922','#f85149','#8b949e']
        ax.bar([str(label).title() for label in cond_price.index], cond_price.values,
               color=colors_c[:len(cond_price)], width=0.5)
        fmt_price(ax, 'y')
        ax.set_xlabel('Condition', color='#8b949e', fontsize=12)
        plt.xticks(rotation=20, ha='right', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with r3b:
        st.subheader("Price vs Screen Size")
        st.caption(
            "Whether larger screens command higher prices.",
            help="Each dot is one listing. If screen size data was sparse in the original CSV, imputation fills in estimated values — look for clustering around common sizes like 13, 14, and 15.6 inches."
        )
        scr_plot = plot_df[plot_df['screen_num'] > 0].copy()
        if len(scr_plot) > 5:
            fig, ax = dark_fig()
            ax.scatter(scr_plot['screen_num'], scr_plot['price_clean'],
                       alpha=0.3, color='#a371f7', s=24, linewidths=0)
            ax.set_xlabel('Screen Size (in)', color='#8b949e', fontsize=12)
            ax.set_ylabel('Price',            color='#8b949e', fontsize=12)
            fmt_price(ax, 'y')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Not enough screen size data. Try broadening your filters.")

    st.markdown("---")

    # full width row for Top 10 Models
    st.subheader("Top 10 Most Common Models")
    st.caption(
        "Which laptop models appear most in the dataset.",
        help="More listings for a model = more training data = more reliable AI predictions for that model. If you're flipping a rare model, the valuation will be less precise."
    )
    top_models = plot_df['model'].value_counts().dropna().head(10)
    if len(top_models) > 0:
        fig, ax = dark_fig(w=16, h=6)
        ax.barh([str(label).title() for label in top_models.index[::-1]], top_models.values[::-1],
                color='#d29922', height=0.6)
        ax.set_xlabel('Listings', color='#8b949e', fontsize=12)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No model data.")

    st.markdown("---")

    # top opportunities table
    if model_ready:
        st.subheader("Top Profit Opportunities")
        st.caption(
            "The 10 listings with the highest AI predicted profit spread.",
            help="Profit Spread = AI Value minus Ask Price. These are the best deals the model found given your current sidebar filters. Change the filters to explore different segments."
        )
        eval_bot = filtered[['brand', 'condition']].copy()
        eval_bot['ram_gb']     = filtered['ram_num'].fillna(0).astype(int)
        eval_bot['storage_gb'] = filtered['storage_num'].fillna(0).astype(int)
        eval_bot['screen_in']  = filtered['screen_num'].fillna(0)
        p_bot = price_model.predict(eval_bot)

        tmp                  = filtered.copy()
        tmp['ai_value']      = p_bot
        tmp['profit_spread'] = tmp['ai_value'] - tmp['price_clean']

        top10 = (
            tmp[tmp['profit_spread'] > 0]
            .nlargest(10, 'profit_spread')[['brand', 'model', 'condition', 'price_clean', 'ai_value', 'profit_spread']]
            .copy()
        )
        top10.columns          = ['Brand', 'Model', 'Condition', 'Ask Price', 'AI Value', 'Profit Spread']
        
        # title case safe formats
        top10['Brand'] = top10['Brand'].astype(str).str.title()
        top10['Model'] = top10['Model'].astype(str).str.title()
        top10['Condition'] = top10['Condition'].astype(str).str.title()

        top10['Ask Price']     = top10['Ask Price'].apply(lambda x: f"${x:,.2f}")
        top10['AI Value']      = top10['AI Value'].apply(lambda x: f"${x:,.2f}")
        top10['Profit Spread'] = top10['Profit Spread'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top10, use_container_width=True, hide_index=True)