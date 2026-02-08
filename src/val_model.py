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

# make engine
conn_str = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(conn_str)

def get_nums(val):
    # grab numbers from text
    if pd.isna(val): return 0
    match = re.search(r'(\d+)', str(val))
    return int(match.group(1)) if match else 0

print("getting data...")
query = "SELECT brand, ram, storage, price_clean FROM raw_ebay_listings WHERE price_clean IS NOT NULL"
df = pd.read_sql(query, engine)

# clean up columns
print("cleaning features...")
df['ram_gb'] = df['ram'].apply(get_nums)
df['storage_gb'] = df['storage'].apply(get_nums)

# remove junk data
df = df[df['price_clean'] > 50] 
df = df[df['price_clean'] < 5000]

# setup inputs
X = df[['brand', 'ram_gb', 'storage_gb']]
y = df['price_clean']

# handle brand text column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['ram_gb', 'storage_gb']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand'])
    ])

# build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# split and train
print("training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# check score
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# save for dashboard
print("saving pickle...")
joblib.dump(model, 'src/laptop_price_model.pkl')
print("done")