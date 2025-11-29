"""Database utilities: engine creation, SQL validation, and execution."""
import time
import pandas as pd
import sqlglot
from sqlalchemy import create_engine, text
from typing import Tuple, Optional


def get_engine(db_url: str):
    """Create SQLAlchemy engine from database URL."""
    return create_engine(db_url, echo=False)


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL query.
    Returns (is_valid, error_message).
    Ensures:
    - SQL is syntactically valid
    - Only SELECT/WITH statements (read-only)
    - Single statement only
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    
    sql_clean = sql.strip().rstrip(';')
    
    # Check if starts with SELECT or WITH
    sql_upper = sql_clean.upper()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, "SQL must start with SELECT or WITH (read-only queries only)"
    
    # Check for dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if f' {keyword} ' in sql_upper or sql_upper.startswith(keyword + ' '):
            return False, f"SQL contains dangerous keyword: {keyword} (read-only queries only)"
    
    # Validate syntax using sqlglot
    try:
        parsed = sqlglot.parse_one(sql_clean, read='sqlite')
        if parsed is None:
            return False, "Failed to parse SQL"
        
        # Check for multiple statements
        statements = sqlglot.parse(sql_clean, read='sqlite')
        if len(statements) > 1:
            return False, "Multiple statements detected (only single statement allowed)"
        
        return True, None
    except Exception as e:
        return False, f"SQL syntax error: {str(e)}"


def execute_sql(
    sql: str,
    engine,
    add_limit: bool = True,
    max_rows: int = 1000
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str], float]:
    """
    Execute SQL query and return results.
    Returns: (success, dataframe, error_message, execution_latency)
    
    Args:
        sql: SQL query string
        engine: SQLAlchemy engine
        add_limit: If True and no LIMIT exists, add LIMIT max_rows
        max_rows: Maximum rows to return (if add_limit=True)
    """
    start_time = time.time()
    
    if not sql or not sql.strip():
        return False, None, "Empty SQL query", 0.0
    
    sql_clean = sql.strip().rstrip(';')
    
    # Add LIMIT if requested and not present
    if add_limit:
        sql_upper = sql_clean.upper()
        if 'LIMIT' not in sql_upper:
            sql_clean = f"{sql_clean} LIMIT {max_rows}"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_clean))
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            latency = time.time() - start_time
            return True, df, None, latency
            
    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        return False, None, error_msg, latency
