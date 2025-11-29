"""
Quick script to set up the adult_income database table.
This fixes the "no such table: adult_income" error.
"""
import sys
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

def setup_database():
    """Set up the adult_income table in the database."""
    print("="*80)
    print("SETTING UP DATABASE")
    print("="*80)
    
    # Find CSV file
    csv_paths = [
        parent_dir.parent / "adult" / "adult.csv",
        parent_dir / "adult.csv",
        Path("adult.csv"),
        Path("adult/adult.csv")
    ]
    
    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break
    
    if not csv_path:
        print("\n[ERROR] adult.csv file not found!")
        print("Please place adult.csv in one of these locations:")
        for path in csv_paths:
            print(f"  - {path}")
        return False
    
    print(f"\n[OK] Found CSV file: {csv_path}")
    
    # Find or create database
    db_path = parent_dir / "income.db"
    if not db_path.exists():
        db_path = parent_dir / "data" / "income.db"
        db_path.parent.mkdir(exist_ok=True)
    
    print(f"[OK] Database path: {db_path}")
    
    # Load CSV
    print(f"\nLoading CSV file...")
    try:
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return False
    
    # Normalize column names
    def normalize_col(col):
        col = str(col).lower().strip()
        col = col.replace(" ", "_")
        col = col.replace("-", "_")
        return col
    
    df.columns = [normalize_col(col) for col in df.columns]
    
    # Column mapping (if needed)
    if "gender" in df.columns:
        df = df.rename(columns={"gender": "sex"})
    if "educational-num" in df.columns or "educational_num" in df.columns:
        df = df.rename(columns={"educational-num": "education_num", "educational_num": "education_num"})
    
    # Clean data
    df = df.replace(['?', '', 'NULL'], pd.NA)
    
    # Create database
    print(f"\nCreating database table...")
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Drop existing table if it exists
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS adult_income"))
            conn.commit()
        
        # Create table
        df.to_sql("adult_income", engine, if_exists="replace", index=False)
        
        # Create indexes
        with engine.connect() as conn:
            for col in ["education", "workclass", "occupation", "marital_status", "income"]:
                try:
                    conn.execute(text(f"CREATE INDEX idx_{col} ON adult_income({col})"))
                except:
                    pass
            conn.commit()
        
        print(f"[OK] Created table 'adult_income' with {len(df)} rows")
        print(f"[OK] Created indexes on key columns")
        
        # Verify
        result = pd.read_sql("SELECT COUNT(*) AS cnt FROM adult_income", engine)
        print(f"[OK] Verification: {result.iloc[0, 0]} rows in table")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create database: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = setup_database()
    if success:
        print("\n" + "="*80)
        print("DATABASE SETUP COMPLETE!")
        print("="*80)
        print("\nYou can now run the RAG comparison script.")
    else:
        print("\n" + "="*80)
        print("DATABASE SETUP FAILED")
        print("="*80)
        print("\nPlease fix the errors above and try again.")

