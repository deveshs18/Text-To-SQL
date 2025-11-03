"""Evaluation metrics: Exact Match (EM) and Execution Accuracy (EX)."""
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sqlalchemy import create_engine
from db import execute_sql


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, remove extra whitespace."""
    sql = sql.strip().lower()
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()


def exact_match(gold_sql: str, generated_sql: str) -> bool:
    """
    Compute Exact Match (EM): normalized SQL strings must be identical.
    """
    gold_norm = normalize_sql(gold_sql)
    gen_norm = normalize_sql(generated_sql)
    return gold_norm == gen_norm


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8, ignore_column_names: bool = False) -> bool:
    """
    Compare two DataFrames order-agnostic and NA-safe.
    Sort columns, sort rows, compare.
    Handles floating point precision and type differences (int vs float).
    
    Args:
        ignore_column_names: If True, compare data values only, ignoring column name differences.
    """
    if df1 is None or df2 is None:
        return False
    
    if df1.empty and df2.empty:
        return True
    
    if df1.empty or df2.empty:
        return False
    
    # Sort columns
    df1_sorted = df1.sort_index(axis=1)
    df2_sorted = df2.sort_index(axis=1)
    
    # Check column names match (unless ignoring column names)
    if not ignore_column_names:
        if list(df1_sorted.columns) != list(df2_sorted.columns):
            return False
    
    # Check same number of columns
    if len(df1_sorted.columns) != len(df2_sorted.columns):
        return False
    
    # Sort rows (convert to list of tuples for sorting)
    try:
        df1_rows = [tuple(row) for row in df1_sorted.values]
        df2_rows = [tuple(row) for row in df2_sorted.values]
        
        df1_rows_sorted = sorted(df1_rows)
        df2_rows_sorted = sorted(df2_rows)
        
        if len(df1_rows_sorted) != len(df2_rows_sorted):
            return False
        
        # Compare row by row with NA-safe and floating-point-aware comparison
        for r1, r2 in zip(df1_rows_sorted, df2_rows_sorted):
            if len(r1) != len(r2):
                return False
            for v1, v2 in zip(r1, r2):
                # Handle NA/None
                if pd.isna(v1) and pd.isna(v2):
                    continue
                if pd.isna(v1) or pd.isna(v2):
                    return False
                
                # Handle numeric comparisons (int vs float, floating point precision)
                try:
                    # Try converting both to float for numeric comparison
                    v1_float = float(v1)
                    v2_float = float(v2)
                    # Use numpy's isclose for floating point comparison
                    if not np.isclose(v1_float, v2_float, rtol=rtol, atol=atol, equal_nan=True):
                        return False
                    continue
                except (ValueError, TypeError):
                    # Not numeric, use exact equality
                    pass
                
                # For non-numeric, use exact equality
                if v1 != v2:
                    return False
        
        return True
    except Exception as e:
        # Fallback: string comparison of sorted CSV (normalize numeric formats)
        try:
            csv1 = df1_sorted.to_csv(index=False)
            csv2 = df2_sorted.to_csv(index=False)
            return csv1 == csv2
        except:
            return False


def execution_accuracy(
    gold_sql: str,
    generated_sql: str,
    engine,
    max_rows: int = 1000
) -> Tuple[bool, Optional[str]]:
    """
    Compute Execution Accuracy (EX): execute both queries and compare results.
    Return (is_match, error_message).
    """
    try:
        # Execute gold SQL
        gold_success, gold_df, gold_err, _ = execute_sql(
            gold_sql, engine, add_limit=False, max_rows=max_rows
        )
        if not gold_success:
            return False, f"Gold SQL execution failed: {gold_err}"
        
        # Execute generated SQL
        gen_success, gen_df, gen_err, _ = execute_sql(
            generated_sql, engine, add_limit=False, max_rows=max_rows
        )
        if not gen_success:
            return False, f"Generated SQL execution failed: {gen_err}"
        
        # Compare results (ignore column names - only compare data values)
        match = compare_dataframes(gold_df, gen_df, ignore_column_names=True)
        return match, None
        
    except Exception as e:
        return False, f"Comparison error: {str(e)}"

