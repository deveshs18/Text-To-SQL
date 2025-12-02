"""
Setup script for MySQL database table: adult_income
Replaces SQLite version with MySQL for faster queries & RAG usage.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------------
# MySQL CONFIGURATION
# -------------------------------
MYSQL_USER = "root"
MYSQL_PASSWORD = "test"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "adultdb"      # Ensure you created this: CREATE DATABASE adultdb;

def setup_database():
    print("="*80)
    print("SETTING UP MySQL DATABASE")
    print("="*80)

    # ---------- Find CSV ----------
    csv_paths = [
        Path("adult.csv"),
        Path("adult/adult.csv"),
        Path("../adult/adult.csv"),
        Path("../../adult/adult.csv")
    ]

    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break

    if not csv_path:
        print("\n[ERROR] adult.csv file not found!")
        print("Place adult.csv in the project folder.")
        return False

    print(f"[OK] Found CSV file: {csv_path}")

    # ---------- Load CSV ----------
    try:
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df)} rows")
    except Exception as e:
        print(f"[ERROR] Failed reading CSV: {e}")
        return False

    # Normalize column names
    def normalize(col):
        col = str(col).lower().strip()
        col = col.replace(" ", "_")
        col = col.replace("-", "_")
        return col

    df.columns = [normalize(c) for c in df.columns]

    # Fix alternate names
    if "gender" in df.columns:
        df = df.rename(columns={"gender": "sex"})
    if "educational-num" in df.columns:
        df = df.rename(columns={"educational-num": "education_num"})

    df = df.replace(["?", "", "NULL"], pd.NA)

    # ---------- Connect to MySQL ----------
    print("\nConnecting to MySQL...")
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}",
        echo=False,
        pool_recycle=3600,
    )

    # ---------- Drop & recreate table ----------
    print("Dropping old table (if exists)...")
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS adult_income"))
            conn.commit()
    except Exception as e:
        print("[ERROR] Could not drop old table:", e)

    print("Creating new table and inserting data (fast mode)...")

    try:
        df.to_sql(
            "adult_income",
            engine,
            if_exists="replace",
            index=False,
            chunksize=500,
            method="multi"
        )
    except Exception as e:
        print("[ERROR] Failed to write to MySQL:", e)
        return False

    print("[OK] Inserted data into MySQL")

    # ---------- Create Indexes ----------
    print("\nCreating indexes...")
    index_cols = ["education", "workclass", "occupation", "marital_status", "income"]

    try:
        with engine.connect() as conn:
            for col in index_cols:
                try:
                    conn.execute(text(f"CREATE INDEX idx_{col} ON adult_income({col})"))
                except Exception:
                    pass
            conn.commit()
    except Exception as e:
        print("[ERROR] Failed creating indexes:", e)

    print("[OK] Indexes created")

    # ---------- Verify ----------
    try:
        result = pd.read_sql("SELECT COUNT(*) AS cnt FROM adult_income", engine)
        print(f"[OK] Verification: {result.iloc[0, 0]} rows present.")
    except:
        print("[ERROR] Verification failed.")

    print("\nDATABASE SETUP COMPLETE!")
    return True


if __name__ == "__main__":
    setup_database()
