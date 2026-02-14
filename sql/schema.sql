DROP TABLE IF EXISTS raw_ebay_listings;

CREATE TABLE raw_ebay_listings (
    id SERIAL PRIMARY KEY,
    brand TEXT,
    model TEXT,
    price_raw VARCHAR(50),      
    price_clean DECIMAL(10, 2),
    condition TEXT,
    processor VARCHAR(255),
    screen_size VARCHAR(100),
    ram VARCHAR(100),
    storage VARCHAR(100),
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);