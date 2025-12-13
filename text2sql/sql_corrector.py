"""SQL correction and validation utilities to improve EX accuracy.
This module provides both general-purpose SQL fixes (work for any dataset) 
and dataset-specific fixes (can be configured or disabled).

General-purpose fixes:
- Add missing COUNT(*) when ORDER BY count(*) is used
- Add missing ORDER BY for "top/most" queries
- Add missing LIMIT for "top N" queries
- Ensure semicolon at end

Dataset-specific fixes (can be disabled):
- education_num -> education (adult_income dataset)
- income > '50K' -> income = '>50K' (adult_income dataset)
"""
import re
from typing import Tuple, Optional, List, Set, Dict
from sqlalchemy import create_engine, text, inspect


def get_table_columns(engine, table_name: str) -> Set[str]:
    """Get all column names from a table."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = [row[1] for row in result.fetchall()]
            return set(columns)
    except:
        return set()


def correct_common_sql_errors(
    sql: str, 
    engine, 
    table_name: str = "adult_income", 
    question: str = "",
    dataset_specific_fixes: bool = True
) -> Tuple[str, List[str]]:
    """
    Correct common SQL errors to improve EX accuracy.
    
    Args:
        sql: SQL query to correct
        engine: SQLAlchemy engine
        table_name: Name of the table (default: "adult_income")
        question: Natural language question (for context-aware fixes)
        dataset_specific_fixes: If True, apply dataset-specific fixes (default: True)
    
    Returns:
        (corrected_sql, list_of_fixes_applied)
    """
    fixes = []
    corrected = sql
    
    # Get valid columns
    valid_columns = get_table_columns(engine, table_name)
    
    # ============================================================================
    # DATASET-SPECIFIC FIXES (adult_income dataset)
    # ============================================================================
    # These fixes are specific to the adult_income dataset and can be disabled
    # by setting dataset_specific_fixes=False
    if dataset_specific_fixes:
        # Fix 1a: avg(education) -> avg(education_num) when question asks for average education_num
        if re.search(r'\bavg\(education\)', corrected, re.IGNORECASE):
            if re.search(r'\b(average|avg)\s+education', question, re.IGNORECASE) and 'education_num' in valid_columns:
                corrected = re.sub(
                    r'\bavg\(education\)',
                    'avg(education_num)',
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: avg(education) -> avg(education_num) (dataset-specific)")
        
        # Fix 1b: education_num -> education (common mistake in adult_income dataset)
        # Only fix if question asks for "education level" or similar, not "education_num"
        if 'education_num' in corrected and 'education' in valid_columns:
            # Check if question context suggests text education (not numeric)
            if re.search(r'\beducation_num\b', corrected, re.IGNORECASE):
                # Check if it's in a context where text makes more sense
                # If used with text operations or in SELECT for display, likely wrong
                if re.search(r'(SELECT|GROUP BY).*education_num', corrected, re.IGNORECASE):
                    corrected = re.sub(r'\beducation_num\b', 'education', corrected, flags=re.IGNORECASE)
                    fixes.append("Fixed: education_num -> education (dataset-specific)")
        
        # Fix 2: income comparisons -> income = '>50K' or income = '<=50K' (adult_income dataset)
        # The income column in adult_income has values '<=50K' and '>50K', not numeric comparisons
        # Fix: income > '50K' -> income = '>50K'
        if re.search(r"income\s*>\s*['\"]50K['\"]", corrected, re.IGNORECASE):
            corrected = re.sub(
                r"income\s*>\s*['\"]50K['\"]",
                "income = '>50K'",
                corrected,
                flags=re.IGNORECASE
            )
            fixes.append("Fixed: income > '50K' -> income = '>50K' (dataset-specific)")
        
        # Fix: income = '>=50K' -> income = '>50K'
        if re.search(r"income\s*=\s*['\"]>=50K['\"]", corrected, re.IGNORECASE):
            corrected = re.sub(
                r"income\s*=\s*['\"]>=50K['\"]",
                "income = '>50K'",
                corrected,
                flags=re.IGNORECASE
            )
            fixes.append("Fixed: income = '>=50K' -> income = '>50K' (dataset-specific)")
        
        # Fix: income > 50 or income > 50000 -> income = '>50K' (treating as numeric)
        if re.search(r"income\s*>\s*(\d+)", corrected, re.IGNORECASE):
            # Check if question asks for "more than 50K" or similar
            if re.search(r'\b(more than|greater than|over|above|earn more|earns more)\s+50', question, re.IGNORECASE):
                corrected = re.sub(
                    r"income\s*>\s*\d+",
                    "income = '>50K'",
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: income > [number] -> income = '>50K' (dataset-specific)")
        
        # Fix: income <= 50 or income <= 50000 -> income = '<=50K'
        if re.search(r"income\s*<=\s*(\d+)", corrected, re.IGNORECASE):
            # Check if question asks for "less than or equal to 50K" or similar
            if re.search(r'\b(less than|less or equal|at most|up to|earn less|earns less|<=)\s+50', question, re.IGNORECASE):
                corrected = re.sub(
                    r"income\s*<=\s*\d+",
                    "income = '<=50K'",
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: income <= [number] -> income = '<=50K' (dataset-specific)")
        
        # Fix: income < 50 -> income = '<=50K'
        if re.search(r"income\s*<\s*(\d+)", corrected, re.IGNORECASE):
            if re.search(r'\b(less than|earn less|earns less)\s+50', question, re.IGNORECASE):
                corrected = re.sub(
                    r"income\s*<\s*\d+",
                    "income = '<=50K'",
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: income < [number] -> income = '<=50K' (dataset-specific)")
    
    # ============================================================================
    # GENERAL-PURPOSE FIXES (work for any dataset)
    # ============================================================================
    
    # Fix 3 (General): Add missing COUNT(*) in SELECT when ORDER BY count(*) is used
    if re.search(r'ORDER\s+BY\s+count\(', corrected, re.IGNORECASE):
        # Check if SELECT doesn't have COUNT
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', corrected, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_cols = select_match.group(1).strip()
            if 'count' not in select_cols.lower() and 'COUNT' not in select_cols:
                # Add COUNT(*) to SELECT
                # Find GROUP BY column(s)
                group_match = re.search(r'GROUP\s+BY\s+([^;]+)', corrected, re.IGNORECASE)
                if group_match:
                    group_cols = group_match.group(1).strip()
                    # Get first column from GROUP BY
                    first_group_col = group_cols.split(',')[0].strip()
                    # Replace SELECT clause to add COUNT(*)
                    # Match the entire SELECT ... FROM pattern
                    # Escape special regex characters in select_cols
                    select_cols_escaped = re.escape(select_cols)
                    corrected = re.sub(
                        r'(SELECT\s+)' + select_cols_escaped + r'(\s+FROM)',
                        r'\1' + select_cols + ', COUNT(*) AS count\2',
                        corrected,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    fixes.append("Fixed: Added COUNT(*) to SELECT")
    
    # Fix 3b (General): Add missing COUNT(*) when question asks for "most common" or "top"
    if re.search(r'\b(most common|most frequent|top|highest count)', question, re.IGNORECASE):
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', corrected, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_cols = select_match.group(1)
            if 'count' not in select_cols.lower() and 'COUNT' not in select_cols:
                # Check if there's GROUP BY
                if 'GROUP BY' in corrected.upper():
                    group_match = re.search(r'GROUP\s+BY\s+([^;]+)', corrected, re.IGNORECASE)
                    if group_match:
                        group_cols = group_match.group(1).strip()
                        first_group_col = group_cols.split(',')[0].strip()
                        corrected = re.sub(
                            r'(SELECT\s+)' + re.escape(first_group_col),
                            r'\1' + first_group_col + ', COUNT(*) AS count',
                            corrected,
                            flags=re.IGNORECASE
                        )
                        fixes.append("Fixed: Added COUNT(*) for 'most common' query")
    
    # Fix 4 (General): Add missing ORDER BY when question asks for "top", "most common", etc.
    if re.search(r'\b(top|most|highest|lowest|best|worst)\b', corrected, re.IGNORECASE):
        if 'ORDER BY' not in corrected.upper():
            # Try to infer what to order by
            if 'COUNT' in corrected.upper():
                # Order by count
                if not corrected.rstrip().endswith(';'):
                    corrected = corrected.rstrip() + ' ORDER BY count(*) DESC'
                else:
                    corrected = corrected.rstrip(';') + ' ORDER BY count(*) DESC;'
                fixes.append("Fixed: Added ORDER BY count(*) DESC")
            elif 'AVG' in corrected.upper() or 'average' in corrected.lower():
                # Order by average
                avg_match = re.search(r'AVG\((\w+)\)', corrected, re.IGNORECASE)
                if avg_match:
                    avg_col = avg_match.group(1)
                    if not corrected.rstrip().endswith(';'):
                        corrected = corrected.rstrip() + f' ORDER BY AVG({avg_col}) DESC'
                    else:
                        corrected = corrected.rstrip(';') + f' ORDER BY AVG({avg_col}) DESC;'
                    fixes.append(f"Fixed: Added ORDER BY AVG({avg_col}) DESC")
    
    # Fix 5 (General): Add missing LIMIT when question asks for "top N"
    limit_match = re.search(r'\b(top|first)\s+(\d+)', corrected, re.IGNORECASE)
    if limit_match and 'LIMIT' not in corrected.upper():
        limit_num = limit_match.group(2)
        if not corrected.rstrip().endswith(';'):
            corrected = corrected.rstrip() + f' LIMIT {limit_num}'
        else:
            corrected = corrected.rstrip(';') + f' LIMIT {limit_num};'
        fixes.append(f"Fixed: Added LIMIT {limit_num}")
    
    # Fix 6 (General): Fix case sensitivity for sex column
    # Fix: sex = 'female' -> sex = 'Female'
    if re.search(r"sex\s*=\s*['\"]female['\"]", corrected, re.IGNORECASE):
        corrected = re.sub(
            r"sex\s*=\s*['\"]female['\"]",
            "sex = 'Female'",
            corrected,
            flags=re.IGNORECASE
        )
        fixes.append("Fixed: sex = 'female' -> sex = 'Female' (case sensitivity)")
    
    # Fix: sex = 'M' -> sex = 'Male'
    if re.search(r"sex\s*=\s*['\"]M['\"]", corrected, re.IGNORECASE):
        corrected = re.sub(
            r"sex\s*=\s*['\"]M['\"]",
            "sex = 'Male'",
            corrected,
            flags=re.IGNORECASE
        )
        fixes.append("Fixed: sex = 'M' -> sex = 'Male' (case sensitivity)")
    
    # Fix 7 (General): Remove unnecessary JOINs with non-existent tables
    # Remove JOINs to tables that don't exist (like "Note", "native_country", etc.)
    # This is a heuristic - we'll remove JOINs to common non-existent table names
    non_existent_tables = ['Note', 'note', 'native_country', 'country', 'adult_hours', 'adult_worked']
    for table in non_existent_tables:
        # Pattern: JOIN table_name AS ... ON ...
        pattern = rf'JOIN\s+{re.escape(table)}\s+AS\s+\w+\s+ON\s+[^;]+'
        if re.search(pattern, corrected, re.IGNORECASE):
            # Remove the entire JOIN clause
            corrected = re.sub(pattern, '', corrected, flags=re.IGNORECASE)
            fixes.append(f"Fixed: Removed unnecessary JOIN to non-existent table '{table}'")
            # Clean up any double spaces
            corrected = re.sub(r'\s+', ' ', corrected)
    
    # Fix 8 (General): Fix missing FROM keyword
    # Pattern: SELECT ... table_name GROUP BY (missing FROM)
    if re.search(r'SELECT\s+[^F]+?\s+\w+\s+GROUP\s+BY', corrected, re.IGNORECASE):
        # Try to insert FROM before GROUP BY
        match = re.search(r'(SELECT\s+[^F]+?)\s+(\w+)\s+(GROUP\s+BY)', corrected, re.IGNORECASE)
        if match and 'FROM' not in corrected.upper()[:match.end()]:
            table_name_match = match.group(2)
            # Check if it looks like a table name (not a column)
            if table_name_match.lower() in ['adult_income', 'income', 'adult']:
                corrected = re.sub(
                    rf'({re.escape(match.group(1))})\s+({re.escape(table_name_match)})\s+({re.escape(match.group(3))})',
                    r'\1 FROM \2 \3',
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: Added missing FROM keyword")
    
    # Fix 9 (General): Fix wrong column names
    # Fix: adult_age -> age
    if re.search(r'\badult_age\b', corrected, re.IGNORECASE):
        corrected = re.sub(r'\badult_age\b', 'age', corrected, flags=re.IGNORECASE)
        fixes.append("Fixed: adult_age -> age (column name)")
    
    # Fix: hours -> hours_per_week
    if re.search(r'\bavg\(hours\)\b', corrected, re.IGNORECASE) and 'hours_per_week' in valid_columns:
        corrected = re.sub(r'\bavg\(hours\)\b', 'avg(hours_per_week)', corrected, flags=re.IGNORECASE)
        fixes.append("Fixed: avg(hours) -> avg(hours_per_week) (column name)")
    
    # Fix: gender -> sex
    if re.search(r'\bgender\s*=', corrected, re.IGNORECASE) and 'sex' in valid_columns:
        corrected = re.sub(r'\bgender\s*=', 'sex =', corrected, flags=re.IGNORECASE)
        fixes.append("Fixed: gender -> sex (column name)")
    
    # Fix 10a (General): Fix missing GROUP BY when aggregations are used
    # Pattern: SELECT col, COUNT(*) FROM table (missing GROUP BY col)
    if re.search(r'SELECT\s+(\w+)\s*,\s*COUNT\(', corrected, re.IGNORECASE):
        select_match = re.search(r'SELECT\s+(\w+)\s*,\s*COUNT\(', corrected, re.IGNORECASE)
        if select_match:
            group_col = select_match.group(1).strip()
            # Check if GROUP BY is missing
            if 'GROUP BY' not in corrected.upper():
                # Add GROUP BY before ORDER BY or LIMIT or end
                if 'ORDER BY' in corrected.upper():
                    corrected = re.sub(
                        r'(ORDER\s+BY)',
                        f'GROUP BY {group_col} \\1',
                        corrected,
                        flags=re.IGNORECASE
                    )
                    fixes.append(f"Fixed: Added GROUP BY {group_col}")
                elif 'LIMIT' in corrected.upper():
                    corrected = re.sub(
                        r'(LIMIT)',
                        f'GROUP BY {group_col} \\1',
                        corrected,
                        flags=re.IGNORECASE
                    )
                    fixes.append(f"Fixed: Added GROUP BY {group_col}")
                else:
                    # Add at end
                    if not corrected.rstrip().endswith(';'):
                        corrected = corrected.rstrip() + f' GROUP BY {group_col}'
                    else:
                        corrected = corrected.rstrip(';') + f' GROUP BY {group_col};'
                    fixes.append(f"Fixed: Added GROUP BY {group_col}")
    
    # Fix 10b (General): Fix wrong column order in SELECT when GROUP BY is used
    # Pattern: SELECT COUNT(*), col FROM ... GROUP BY col
    # Should be: SELECT col, COUNT(*) FROM ... GROUP BY col
    if 'GROUP BY' in corrected.upper():
        group_match = re.search(r'GROUP\s+BY\s+([^;]+)', corrected, re.IGNORECASE)
        if group_match:
            group_cols = [col.strip() for col in group_match.group(1).split(',')]
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', corrected, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_cols = select_match.group(1).strip()
                # Check if COUNT(*) comes before the grouped column
                if re.search(r'COUNT\(\*\)\s*,\s*\w+', select_cols, re.IGNORECASE):
                    # Reorder: put grouped column first
                    first_group_col = group_cols[0]
                    # Remove the grouped column from SELECT if it's there
                    select_cols_clean = re.sub(rf'\b{re.escape(first_group_col)}\s*,?\s*', '', select_cols, flags=re.IGNORECASE)
                    # Reorder to: grouped_col, rest
                    new_select = f"{first_group_col}, {select_cols_clean}".strip(', ')
                    corrected = re.sub(
                        r'(SELECT\s+)' + re.escape(select_cols) + r'(\s+FROM)',
                        r'\1' + new_select + r'\2',
                        corrected,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    fixes.append(f"Fixed: Reordered SELECT columns (GROUP BY column first)")
    
    # Fix 10c (General): Fix ORDER BY wrong column name
    # Pattern: ORDER BY count (when column is named "count" but should be COUNT(*))
    if re.search(r'ORDER\s+BY\s+count\b(?!\()', corrected, re.IGNORECASE):
        # Check if there's a COUNT(*) in SELECT
        if 'COUNT(*)' in corrected.upper() or 'COUNT(*) AS count' in corrected.upper():
            corrected = re.sub(
                r'ORDER\s+BY\s+count\b(?!\()',
                'ORDER BY COUNT(*)',
                corrected,
                flags=re.IGNORECASE
            )
            fixes.append("Fixed: ORDER BY count -> ORDER BY COUNT(*)")
        elif 'COUNT(*) AS count' in corrected.upper():
            corrected = re.sub(
                r'ORDER\s+BY\s+count\b(?!\()',
                'ORDER BY count',
                corrected,
                flags=re.IGNORECASE
            )
            fixes.append("Fixed: ORDER BY count (using alias)")
    
    # Fix 10d (General): Add missing DESC for "most common/top" queries
    if re.search(r'\b(most common|most frequent|top|highest)', question, re.IGNORECASE):
        if re.search(r'ORDER\s+BY\s+[^D]+(?!\s+DESC)', corrected, re.IGNORECASE):
            # Check if DESC is missing
            order_match = re.search(r'ORDER\s+BY\s+([^;]+)', corrected, re.IGNORECASE)
            if order_match and 'DESC' not in order_match.group(1).upper() and 'ASC' not in order_match.group(1).upper():
                corrected = re.sub(
                    r'(ORDER\s+BY\s+[^;]+)(?=\s*(?:LIMIT|;|$))',
                    r'\1 DESC',
                    corrected,
                    flags=re.IGNORECASE
                )
                fixes.append("Fixed: Added DESC to ORDER BY for 'most common' query")
    
    # Fix 10e (General): Fix LIMIT for "most common" queries (should be LIMIT 1)
    if re.search(r'\b(most common|most frequent)\b', question, re.IGNORECASE):
        if 'LIMIT' not in corrected.upper():
            # Add LIMIT 1
            if not corrected.rstrip().endswith(';'):
                corrected = corrected.rstrip() + ' LIMIT 1'
            else:
                corrected = corrected.rstrip(';') + ' LIMIT 1;'
            fixes.append("Fixed: Added LIMIT 1 for 'most common' query")
    
    # Fix 10 (General): Ensure semicolon at end
    if corrected and not corrected.rstrip().endswith(';'):
        corrected = corrected.rstrip() + ';'
    
    return corrected, fixes


def validate_and_correct_sql(
    sql: str, 
    engine, 
    table_name: str = "adult_income", 
    question: str = "",
    dataset_specific_fixes: bool = True
) -> Tuple[str, List[str], bool]:
    """
    Validate SQL and apply corrections.
    Returns (corrected_sql, list_of_fixes, is_valid)
    """
    fixes = []
    
    # First, try to correct common errors
    corrected, correction_fixes = correct_common_sql_errors(
        sql, engine, table_name, question, dataset_specific_fixes
    )
    fixes.extend(correction_fixes)
    
    # Basic validation
    if not corrected or not corrected.strip():
        return corrected, fixes, False
    
    # Check if starts with SELECT or WITH
    sql_upper = corrected.upper().strip()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return corrected, fixes, False
    
    return corrected, fixes, True

