"""Advanced evaluation metrics for Text-to-SQL comparison."""
import re
import sqlglot
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import Counter
import math
from db import execute_sql


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, remove extra whitespace."""
    sql = sql.strip().lower()
    sql = sql.rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()


def exact_match(gold_sql: str, generated_sql: str) -> bool:
    """Exact Match (EM): normalized SQL strings must be identical."""
    gold_norm = normalize_sql(gold_sql)
    gen_norm = normalize_sql(generated_sql)
    return gold_norm == gen_norm


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8, ignore_column_names: bool = False) -> bool:
    """Compare two DataFrames order-agnostic and NA-safe. Handles floating point precision and type differences."""
    if df1 is None or df2 is None:
        return False
    if df1.empty and df2.empty:
        return True
    if df1.empty or df2.empty:
        return False
    
    df1_sorted = df1.sort_index(axis=1)
    df2_sorted = df2.sort_index(axis=1)
    
    # Check column names match (unless ignoring column names)
    if not ignore_column_names:
        if list(df1_sorted.columns) != list(df2_sorted.columns):
            return False
    
    # Check same number of columns
    if len(df1_sorted.columns) != len(df2_sorted.columns):
        return False
    
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
    except Exception:
        try:
            return df1_sorted.to_csv() == df2_sorted.to_csv()
        except:
            return False


def execution_accuracy(
    gold_sql: str,
    generated_sql: str,
    engine,
    max_rows: int = 1000
) -> Tuple[bool, Optional[str]]:
    """Execution Accuracy (EX): execute both queries and compare results."""
    try:
        gold_success, gold_df, gold_err, _ = execute_sql(
            gold_sql, engine, add_limit=False, max_rows=max_rows
        )
        if not gold_success:
            return False, f"Gold SQL execution failed: {gold_err}"
        
        gen_success, gen_df, gen_err, _ = execute_sql(
            generated_sql, engine, add_limit=False, max_rows=max_rows
        )
        if not gen_success:
            return False, f"Generated SQL execution failed: {gen_err}"
        
        match = compare_dataframes(gold_df, gen_df, ignore_column_names=True)
        return match, None
    except Exception as e:
        return False, f"Comparison error: {str(e)}"


def semantic_match(gold_sql: str, generated_sql: str) -> Tuple[bool, Dict]:
    """
    Semantic Match (SM): Check logical equivalence of SQL operations.
    Returns (is_match, details_dict).
    """
    details = {
        "select_match": False,
        "from_match": False,
        "where_match": False,
        "group_by_match": False,
        "having_match": False,
        "order_by_match": False,
        "limit_match": False,
        "aggregate_match": False
    }
    
    try:
        # Parse both SQL queries
        gold_parsed = sqlglot.parse_one(gold_sql, read='sqlite')
        gen_parsed = sqlglot.parse_one(generated_sql, read='sqlite')
        
        # Extract SELECT columns
        gold_selects = [str(col) for col in gold_parsed.find_all(sqlglot.expressions.Select)]
        gen_selects = [str(col) for col in gen_parsed.find_all(sqlglot.expressions.Select)]
        details["select_match"] = set(gold_selects) == set(gen_selects)
        
        # Extract FROM tables
        gold_tables = [str(table) for table in gold_parsed.find_all(sqlglot.expressions.Table)]
        gen_tables = [str(table) for table in gen_parsed.find_all(sqlglot.expressions.Table)]
        details["from_match"] = set(gold_tables) == set(gen_tables)
        
        # Check WHERE clauses
        gold_where = gold_parsed.find(sqlglot.expressions.Where)
        gen_where = gen_parsed.find(sqlglot.expressions.Where)
        details["where_match"] = (gold_where is None and gen_where is None) or \
                                 (gold_where is not None and gen_where is not None)
        
        # Check GROUP BY
        gold_group = gold_parsed.find(sqlglot.expressions.Group)
        gen_group = gen_parsed.find(sqlglot.expressions.Group)
        details["group_by_match"] = (gold_group is None and gen_group is None) or \
                                    (gold_group is not None and gen_group is not None)
        
        # Check ORDER BY
        gold_order = gold_parsed.find(sqlglot.expressions.Order)
        gen_order = gen_parsed.find(sqlglot.expressions.Order)
        details["order_by_match"] = (gold_order is None and gen_order is None) or \
                                    (gold_order is not None and gen_order is not None)
        
        # Check LIMIT
        gold_limit = gold_parsed.find(sqlglot.expressions.Limit)
        gen_limit = gen_parsed.find(sqlglot.expressions.Limit)
        details["limit_match"] = (gold_limit is None and gen_limit is None) or \
                                (gold_limit is not None and gen_limit is not None)
        
        # Check aggregate functions
        gold_aggs = [str(agg) for agg in gold_parsed.find_all(sqlglot.expressions.AggFunc)]
        gen_aggs = [str(agg) for agg in gen_parsed.find_all(sqlglot.expressions.AggFunc)]
        details["aggregate_match"] = set(gold_aggs) == set(gen_aggs)
        
        # Overall semantic match: at least 6/8 components match
        match_count = sum(details.values())
        is_match = match_count >= 6
        
        return is_match, details
        
    except Exception as e:
        # If parsing fails, fall back to basic comparison
        return False, {"error": str(e)}


def tokenize_sql(sql: str) -> List[str]:
    """Tokenize SQL into list of tokens."""
    # Normalize first
    sql = normalize_sql(sql)
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[()\[\],;]|\*|=|!=|<>|<=|>=|<|>', sql)
    return [t.lower() for t in tokens if t.strip()]


def f1_score(gold_sql: str, generated_sql: str) -> Tuple[float, Dict]:
    """
    F1-Score: Token-level precision/recall for SQL.
    Returns (f1_score, details_dict).
    """
    gold_tokens = Counter(tokenize_sql(gold_sql))
    gen_tokens = Counter(tokenize_sql(generated_sql))
    
    # Calculate intersection
    intersection = sum((gold_tokens & gen_tokens).values())
    
    # Precision: how many generated tokens are correct
    gen_total = sum(gen_tokens.values())
    precision = intersection / gen_total if gen_total > 0 else 0.0
    
    # Recall: how many gold tokens are found
    gold_total = sum(gold_tokens.values())
    recall = intersection / gold_total if gold_total > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_tokens": gold_total,
        "gen_tokens": gen_total,
        "common_tokens": intersection
    }


def ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Generate n-grams from tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(gold_sql: str, generated_sql: str, max_n: int = 4) -> Tuple[float, Dict]:
    """
    BLEU Score: N-gram overlap between gold and generated SQL.
    Returns (bleu_score, details_dict).
    """
    gold_tokens = tokenize_sql(gold_sql)
    gen_tokens = tokenize_sql(generated_sql)
    
    if len(gen_tokens) == 0:
        return 0.0, {"error": "Generated SQL has no tokens"}
    
    if len(gold_tokens) == 0:
        return 0.0, {"error": "Gold SQL has no tokens"}
    
    # Calculate precision for each n-gram order
    precisions = []
    details = {}
    
    for n in range(1, max_n + 1):
        gold_ngrams = Counter(ngrams(gold_tokens, n))
        gen_ngrams = Counter(ngrams(gen_tokens, n))
        
        # Count matches
        matches = sum((gold_ngrams & gen_ngrams).values())
        total_gen = sum(gen_ngrams.values())
        
        precision = matches / total_gen if total_gen > 0 else 0.0
        precisions.append(precision)
        details[f"precision_{n}"] = precision
        details[f"matches_{n}"] = matches
        details[f"total_gen_{n}"] = total_gen
    
    # Calculate brevity penalty
    if len(gen_tokens) < len(gold_tokens):
        brevity_penalty = math.exp(1 - len(gold_tokens) / len(gen_tokens))
    else:
        brevity_penalty = 1.0
    
    # BLEU = brevity_penalty * geometric_mean(precisions)
    bleu = brevity_penalty * math.exp(sum(math.log(p) if p > 0 else -10 for p in precisions) / len(precisions))
    
    details["bleu"] = bleu
    details["brevity_penalty"] = brevity_penalty
    
    return bleu, details


def rouge_l_score(gold_sql: str, generated_sql: str) -> Tuple[float, Dict]:
    """
    ROUGE-L Score: Longest Common Subsequence (LCS) based similarity.
    Returns (rouge_l_score, details_dict).
    """
    gold_tokens = tokenize_sql(gold_sql)
    gen_tokens = tokenize_sql(generated_sql)
    
    if len(gold_tokens) == 0 or len(gen_tokens) == 0:
        return 0.0, {"error": "Empty token sequence"}
    
    # Calculate LCS length
    def lcs_length(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs = lcs_length(gold_tokens, gen_tokens)
    
    # ROUGE-L = F1 of LCS
    precision = lcs / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
    recall = lcs / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, {
        "rouge_l": f1,
        "precision": precision,
        "recall": recall,
        "lcs_length": lcs,
        "gold_length": len(gold_tokens),
        "gen_length": len(gen_tokens)
    }


def calculate_all_metrics(
    gold_sql: str,
    generated_sql: str,
    engine,
    include_execution: bool = True
) -> Dict:
    """
    Calculate all metrics at once.
    Returns dictionary with all metric values.
    """
    metrics = {
        "em": exact_match(gold_sql, generated_sql),
        "semantic_match": False,
        "semantic_details": {},
        "f1_score": 0.0,
        "f1_details": {},
        "bleu_score": 0.0,
        "bleu_details": {},
        "rouge_l_score": 0.0,
        "rouge_l_details": {},
        "execution_accuracy": False,
        "execution_error": None
    }
    
    # Semantic Match
    try:
        sm_match, sm_details = semantic_match(gold_sql, generated_sql)
        metrics["semantic_match"] = sm_match
        metrics["semantic_details"] = sm_details
    except Exception as e:
        metrics["semantic_details"] = {"error": str(e)}
    
    # F1 Score
    try:
        f1, f1_details = f1_score(gold_sql, generated_sql)
        metrics["f1_score"] = f1
        metrics["f1_details"] = f1_details
    except Exception as e:
        metrics["f1_details"] = {"error": str(e)}
    
    # BLEU Score
    try:
        bleu, bleu_details = bleu_score(gold_sql, generated_sql)
        metrics["bleu_score"] = bleu
        metrics["bleu_details"] = bleu_details
    except Exception as e:
        metrics["bleu_details"] = {"error": str(e)}
    
    # ROUGE-L Score
    try:
        rouge_l, rouge_l_details = rouge_l_score(gold_sql, generated_sql)
        metrics["rouge_l_score"] = rouge_l
        metrics["rouge_l_details"] = rouge_l_details
    except Exception as e:
        metrics["rouge_l_details"] = {"error": str(e)}
    
    # Execution Accuracy (if engine provided)
    if include_execution and engine:
        try:
            ex_match, ex_error = execution_accuracy(gold_sql, generated_sql, engine)
            metrics["execution_accuracy"] = ex_match
            metrics["execution_error"] = ex_error
        except Exception as e:
            metrics["execution_error"] = str(e)
    
    return metrics

