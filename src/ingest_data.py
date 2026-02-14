import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import os

# db settings
db_user = "postgres"
db_pass = "1234"
db_host = "localhost"
db_port = "5432"
db_name = "marketlens"

conn_str = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

def fix_price(price_str):
    # clean price string
    if pd.isna(price_str): return None
    p = str(price_str).split(" to ")[0]
    clean = p.replace("$", "").replace(",", "")
    try:
        return float(clean)
    except:
        return None

def main():
    print("starting import...")
    
    # locate files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    csv_path = os.path.join(root, 'data', 'raw_training_data.csv')
    schema_path = os.path.join(root, 'sql', 'schema.sql')

    if not os.path.exists(csv_path):
        print("error: no csv found")
        return

    engine = create_engine(conn_str)
    
    # reset db structure
    print("resetting table...")
    with engine.connect() as con:
        with open(schema_path, "r") as f:
            con.execute(text(f.read()))
            con.commit()
            
    # load csv
    print("reading csv...")
    df = pd.read_csv(csv_path)

    # map columns
    clean_df = pd.DataFrame()
    clean_df['brand'] = df['Brand']
    clean_df['model'] = df['Model']
    clean_df['price_raw'] = df['Price']
    clean_df['condition'] = df['Condition']
    clean_df['processor'] = df['Processor']
    clean_df['screen_size'] = df['Screen Size']
    clean_df['ram'] = df['Ram Size']
    clean_df['storage'] = df['SSD Capacity']
    
    # clean prices
    print("fixing prices...")
    clean_df['price_clean'] = clean_df['price_raw'].apply(fix_price)
    
    # save to sql
    print("uploading to db...")
    clean_df.to_sql('raw_ebay_listings', engine, if_exists='append', index=False)
    print("done")

if __name__ == "__main__":
    main()