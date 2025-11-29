"""Generic script to load any CSV file into SQLite database."""
import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def normalize_column_name(col: str) -> str:
    """Normalize column names: lowercase, spaces->_, -->_."""
    col = str(col).lower().strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "_")
    return col


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: normalize headers, handle missing values."""
    # Normalize column names
    df.columns = [normalize_column_name(col) for col in df.columns]
    
    # Replace common missing value indicators
    df = df.replace(['?', '', 'NULL', 'null', 'None'], pd.NA)
    
    # Trim string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('', pd.NA).replace('nan', pd.NA)
    
    return df


def load_csv_to_db(
    csv_path: str,
    db_path: str,
    table_name: str = None,
    max_rows: int = None
):
    """
    Load CSV file into SQLite database.
    
    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database (e.g., "mydata.db")
        table_name: Name for the table (defaults to CSV filename without extension)
        max_rows: Maximum rows to load (None = all rows)
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Determine table name
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        table_name = normalize_column_name(table_name)
    
    print(f"📂 Loading CSV: {csv_path}")
    print(f"📊 Table name: {table_name}")
    print(f"💾 Database: {db_path}")
    
    # Read CSV
    try:
        if max_rows:
            print(f"📝 Loading first {max_rows} rows...")
            df = pd.read_csv(csv_path, nrows=max_rows)
        else:
            print("📝 Loading all rows (this may take a moment)...")
            df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Clean data
    print("🧹 Cleaning data...")
    df = clean_dataframe(df)
    
    # Show column info
    print(f"\n📋 Columns: {', '.join(df.columns.tolist())}")
    
    # Create database
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, echo=False)
    
    # Write to database
    try:
        print(f"\n💾 Writing to database...")
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"✅ Created table '{table_name}' with {len(df)} rows")
    except Exception as e:
        print(f"❌ Error writing to database: {e}")
        sys.exit(1)
    
    # Create indexes on common columns (optional)
    try:
        print("\n🔍 Creating indexes...")
        with engine.connect() as conn:
            # Index on first few columns (if they look like IDs or keys)
            for col in df.columns[:5]:
                if len(df[col].unique()) > 10:  # Only index if has reasonable cardinality
                    try:
                        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}({col})"))
                        print(f"  ✓ Indexed: {col}")
                    except:
                        pass
    except Exception as e:
        print(f"⚠️  Warning: Could not create indexes: {e}")
    
    print(f"\n✅ Success! Database '{db_path}' created with table '{table_name}'")
    print(f"\n📝 Next steps:")
    print(f"   1. Update .env file: DB_URL=sqlite:///{db_path}")
    print(f"   2. Run: streamlit run app.py")
    print(f"   3. Or test with: python -c \"from db import get_engine; from schema_retriever import get_all_tables; e=get_engine('sqlite:///{db_path}'); print(get_all_tables(e))\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_csv_to_db.py <csv_path> [db_path] [table_name] [max_rows]")
        print("\nExamples:")
        print("  python load_csv_to_db.py archive/customers.csv customers.db")
        print("  python load_csv_to_db.py archive/orders.csv orders.db orders 1000")
        print("  python load_csv_to_db.py archive/products.csv products.db")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else f"{os.path.splitext(os.path.basename(csv_path))[0]}.db"
    table_name = sys.argv[3] if len(sys.argv) > 3 else None
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    load_csv_to_db(csv_path, db_path, table_name, max_rows)

