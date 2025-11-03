"""Build income.db from Adult Income CSV. Loads all rows by default, or specify max_rows to limit."""
import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def normalize_column_name(col: str) -> str:
    """Normalize column names: lowercase, spaces->_, -->_."""
    col = col.lower().strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    return col


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: normalize headers, strip '?' as NA, trim strings."""
    # Normalize column names
    df.columns = [normalize_column_name(col) for col in df.columns]
    
    # Map column names to match actual DB schema
    column_mapping = {
        'gender': 'sex',
        'educational_num': 'education_num'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Replace '?' with NA
    df = df.replace('?', pd.NA)
    
    # Trim string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        # Convert back to NA if empty
        df[col] = df[col].replace('', pd.NA).replace('nan', pd.NA)
    
    return df


def build_income_db(csv_path: str = None, db_path: str = "income.db", max_rows: int = None):
    """
    Build income.db from CSV.
    If max_rows is None, loads all rows from CSV.
    """
    if csv_path is None:
        # Try common locations
        possible_paths = [
            "adult/adult.csv",
            "adult.csv",
            "../adult/adult.csv",
            "./adult.csv"
        ]
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            print("Error: Could not find adult.csv. Please specify CSV path.")
            print("Usage: python seed_income_db.py [csv_path] [db_path] [max_rows]")
            print("      Set max_rows to 0 or omit to load all rows")
            sys.exit(1)
    
    print(f"Reading CSV from: {csv_path}")
    
    # Read CSV (all rows if max_rows is None or 0)
    try:
        if max_rows is None or max_rows == 0:
            print("Loading ALL rows from CSV (this may take a moment...)")
            df = pd.read_csv(csv_path)
        else:
            print(f"Loading first {max_rows} rows from CSV...")
            df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Clean data
    df = clean_dataframe(df)
    
    # Create SQLite database
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, echo=False)
    
    # Write to database
    table_name = "adult_income"
    try:
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Created table '{table_name}' with {len(df)} rows")
    except Exception as e:
        print(f"Error writing to database: {e}")
        sys.exit(1)
    
    # Create indexes
    index_columns = ["education", "workclass", "occupation", "marital_status", "income"]
    existing_columns = set(df.columns)
    
    with engine.connect() as conn:
        for col in index_columns:
            if col in existing_columns:
                try:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{col} ON {table_name}({col})"))
                    conn.commit()
                    print(f"Created index on {col}")
                except Exception as e:
                    print(f"Warning: Could not create index on {col}: {e}")
    
    # Print sample
    print("\nSample rows:")
    print(df.head(5).to_string())
    print(f"\nRow count: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"\nDatabase created: {db_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    db_path = sys.argv[2] if len(sys.argv) > 2 else "income.db"
    # If max_rows argument is "0" or omitted, load all rows
    if len(sys.argv) > 3:
        max_rows_arg = sys.argv[3]
        max_rows = None if max_rows_arg.lower() in ['0', 'all', 'none'] else int(max_rows_arg)
    else:
        max_rows = None  # Default: load all rows
    
    build_income_db(csv_path, db_path, max_rows)

