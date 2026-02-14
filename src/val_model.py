import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import re

# db connection
db_user = "postgres"
db_pass = "1234" 
db_host = "localhost"
db_port = "5432"
db_name = "marketlens"

conn_str = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(conn_str)

def get_nums(val):
    # extract ints
    if pd.isna(val): return 0
    match = re.search(r'(\d+)', str(val))
    return int(match.group(1)) if match else 0

def get_screen(val):
    # extract float for screen
    if pd.isna(val): return 0.0
    match = re.search(r'(\d+\.?\d*)', str(val))
    return float(match.group(1)) if match else 0.0

if __name__ == "__main__":
    print("getting data...")
    q = "SELECT brand, ram, storage, condition, screen_size, price_clean FROM raw_ebay_listings WHERE price_clean IS NOT NULL"
    df = pd.read_sql(q, engine)

    # clean features
    print("processing features...")
    df['ram_gb'] = df['ram'].apply(get_nums)
    df['storage_gb'] = df['storage'].apply(get_nums)
    df['screen_in'] = df['screen_size'].apply(get_screen)
    
    # filter junk
    df = df[(df['price_clean'] > 50) & (df['price_clean'] < 5000)]
    
    # setup vars
    features = ['brand', 'condition', 'ram_gb', 'storage_gb', 'screen_in']
    X = df[features]
    y = df['price_clean']

    # handle text columns
    # brand/condition are categories, others are numbers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['ram_gb', 'storage_gb', 'screen_in']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'condition'])
        ])

    # build brain
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # train
    print("training...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    print(f"Accuracy: {model.score(X_test, y_test):.2f}")

    # save
    print("saving...")
    joblib.dump(model, 'src/laptop_price_model.pkl')
    print("done")