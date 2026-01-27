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

connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

def fix_price(price_str):
    # helpers to clean up the price strings from ebay
    if pd.isna(price_str):
        return None
        
    price_str = str(price_str)
    
    # some listings have ranges like "300 to 400", just take the first one
    if " to " in price_str:
        price_str = price_str.split(" to ")[0]
        
    # remove the $ and commas
    clean_string = price_str.replace("$", "").replace(",", "")
    
    try:
        return float(clean_string)
    except:
        return None

def main():
    print("Starting import...")
    
    # get the path to this script file so we can find the data folder relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # path to the csv file
    csv_path = os.path.join(project_root, 'data', 'raw_training_data.csv')
    schema_path = os.path.join(project_root, 'sql', 'schema.sql')

    print(f"Looking for data at: {csv_path}")

    if not os.path.exists(csv_path):
        print("Error: Cant find the file. Make sure raw_training_data.csv is in the data folder.")
        return

    # connect to db
    engine = create_engine(connection_string)
    
    print("Resetting table...")
    with engine.connect() as con:
        with open(schema_path, "r") as f:
            sql_command = text(f.read())
            con.execute(sql_command)
            con.commit()
            
    print("Reading csv file...")
    df = pd.read_csv(csv_path)

    # map the csv columns to our database table columns
    clean_df = pd.DataFrame()
    clean_df['brand'] = df['Brand']
    clean_df['model'] = df['Model']
    clean_df['price_raw'] = df['Price']
    clean_df['condition'] = df['Condition']
    clean_df['processor'] = df['Processor']
    clean_df['ram'] = df['Ram Size']
    clean_df['storage'] = df['SSD Capacity']
    
    # run the price cleaning function
    print("Cleaning prices...")
    clean_df['price_clean'] = clean_df['price_raw'].apply(fix_price)
    
    # drop rows where we couldn't get a price
    clean_df = clean_df.dropna(subset=['price_clean'])
    
    print(f"Uploading {len(clean_df)} listings to postgres...")
    clean_df.to_sql('raw_ebay_listings', engine, if_exists='append', index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()