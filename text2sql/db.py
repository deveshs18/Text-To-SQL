"""Database utilities: SQLAlchemy engine, SQL validation, and read-only execution."""
import re
import time
from typing import Tuple, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sqlglot


def get_engine(db_url: str):
    """Create SQLAlchemy engine from DB_URL."""
    return create_engine(db_url, echo=False)


def validate_sql(sql: str, use_sqlglot: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL: must be SELECT or WITH (CTE).
    Reject multiple statements.
    Return (is_valid, error_message).
    """
    sql_orig = sql.strip()
    
    # Remove trailing semicolon
    sql = sql_orig.rstrip(';').strip()
    
    if not sql:
        return False, "Empty SQL statement"
    
    # Case-insensitive check for SELECT or WITH
    sql_upper = sql.upper().strip()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, "Only SELECT and WITH (CTE) statements are allowed"
    
    # Reject multiple statements (check original before removing semicolon)
    if sql_orig.count(';') > 1 or (sql_orig.count(';') == 1 and not sql_orig.rstrip().endswith(';')):
        return False, "Multiple statements not allowed"
    
    # Optional: sqlglot parsing (wrapped in try/except, don't access .kind)
    if use_sqlglot:
        try:
            # Just verify it parses - don't check .kind as it may not exist
            sqlglot.parse_one(sql, read='sqlite')
        except Exception:
            # If parsing fails, fall back to basic check
            # The basic check above already ensures it starts with SELECT or WITH
            pass
    
    return True, None


def add_limit_if_missing(sql: str, limit: int = 100) -> str:
    """Add LIMIT clause if missing, preserving existing LIMIT if present."""
    sql = sql.strip().rstrip(';')
    sql_upper = sql.upper()
    
    # Check if LIMIT already exists
    if 'LIMIT' in sql_upper:
        return sql
    
    # Simple append (works for most cases)
    return f"{sql} LIMIT {limit}"


def execute_sql(
    sql: str,
    engine,
    add_limit: bool = True,
    limit: int = 100,
    max_rows: int = 100
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str], float]:
    """
    Execute SQL query read-only.
    Return (success, dataframe, error_message, latency_sec).
    """
    start_time = time.time()
    
    # Validate first
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return False, None, error, time.time() - start_time
    
    # Add limit if requested
    if add_limit:
        sql = add_limit_if_missing(sql, limit)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Truncate to max_rows for display
            if len(df) > max_rows:
                df = df.head(max_rows)
            
            latency = time.time() - start_time
            return True, df, None, latency
            
    except SQLAlchemyError as e:
        error_msg = str(e)
        latency = time.time() - start_time
        return False, None, error_msg, latency
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        latency = time.time() - start_time
        return False, None, error_msg, latency

