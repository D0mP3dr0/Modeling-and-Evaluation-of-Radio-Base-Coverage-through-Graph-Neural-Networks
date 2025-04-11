import pandas as pd
import os

# Check if file exists
csv_path = 'data/csv_licenciamento_bruto.csv.csv'
if not os.path.exists(csv_path):
    print(f"File does not exist: {csv_path}")
    exit(1)

# Read the CSV file
print(f"Reading file: {csv_path}")
try:
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Print all column names
    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    # Check for coordinate columns
    lat_lon_cols = [col for col in df.columns if 'lat' in col.lower() or 'lon' in col.lower() or 'coordenada' in col.lower()]
    print(f"\nPossible coordinate columns: {lat_lon_cols}")
    
    # Print sample data (first 3 rows)
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Print data types
    print("\nData types:")
    print(df.dtypes)
    
except Exception as e:
    print(f"Error reading CSV: {e}") 