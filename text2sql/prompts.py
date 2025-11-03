"""Prompt builders for Few-Shot (FS), Chain-of-Thought (CoT), Least-to-Most (LtM), and Execution-Guided (EG)."""
from typing import List, Tuple
import re


def extract_table_name_from_schema(schema: str) -> str:
    """Extract table name from schema snippet. Defaults to 'your_table' if not found."""
    # Schema format is typically: "table_name(column1, column2, ...)"
    match = re.match(r'^(\w+)\(', schema.strip())
    if match:
        return match.group(1)
    return "your_table"


def replace_table_name_in_sql(sql: str, old_table: str, new_table: str) -> str:
    """Replace table name in SQL string. Handles FROM, JOIN, etc."""
    # Replace FROM old_table
    sql = re.sub(rf'\bFROM\s+{re.escape(old_table)}\b', f'FROM {new_table}', sql, flags=re.IGNORECASE)
    # Replace JOIN old_table
    sql = re.sub(rf'\bJOIN\s+{re.escape(old_table)}\b', f'JOIN {new_table}', sql, flags=re.IGNORECASE)
    return sql


# Simple examples - using placeholder that will be replaced
SIMPLE_EXAMPLES_TEMPLATE = [
    {
        "question": "Average hours_per_week by education (top 10).",
        "sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM {TABLE} GROUP BY education ORDER BY avg_hours DESC LIMIT 10;"
    },
    {
        "question": "Top 5 occupation by count where income = '>50K'.",
        "sql": "SELECT occupation, COUNT(*) AS count FROM {TABLE} WHERE income = '>50K' GROUP BY occupation ORDER BY count DESC LIMIT 5;"
    },
    {
        "question": "Count by race and sex.",
        "sql": "SELECT race, sex, COUNT(*) AS count FROM {TABLE} GROUP BY race, sex ORDER BY count DESC LIMIT 100;"
    },
    {
        "question": "How many people have sex = 'Female'?",
        "sql": "SELECT COUNT(*) AS count FROM {TABLE} WHERE sex = 'Female';"
    },
]


# Complex examples with CTEs and window functions - using placeholder
COMPLEX_EXAMPLES_TEMPLATE = [
    {
        "question": "For each education level, show count, percentage earning more than 50K, and the most common occupation.",
        "sql": """WITH occ_counts AS (
  SELECT education, occupation, COUNT(*) AS occ_cnt
  FROM {TABLE}
  GROUP BY education, occupation
),
top_occ AS (
  SELECT education, occupation
  FROM (
    SELECT education, occupation, occ_cnt,
           ROW_NUMBER() OVER (PARTITION BY education ORDER BY occ_cnt DESC, occupation ASC) AS rn
    FROM occ_counts
  )
  WHERE rn = 1
),
agg AS (
  SELECT
    education,
    COUNT(*) AS total_people,
    AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income
  FROM {TABLE}
  GROUP BY education
)
SELECT
  a.education,
  a.total_people,
  ROUND(a.pct_high_income * 100.0, 2) AS pct_high_income_percent,
  t.occupation AS top_occupation
FROM agg a
LEFT JOIN top_occ t ON a.education = t.education
ORDER BY a.pct_high_income DESC, a.total_people DESC
LIMIT 100;"""
    },
    {
        "question": "For each race and sex combination, show total count, average age, and the most frequent workclass.",
        "sql": """WITH workclass_counts AS (
  SELECT race, sex, workclass, COUNT(*) AS wc_cnt
  FROM {TABLE}
  GROUP BY race, sex, workclass
),
top_workclass AS (
  SELECT race, sex, workclass
  FROM (
    SELECT race, sex, workclass, wc_cnt,
           ROW_NUMBER() OVER (PARTITION BY race, sex ORDER BY wc_cnt DESC, workclass ASC) AS rn
    FROM workclass_counts
  )
  WHERE rn = 1
),
agg AS (
  SELECT
    race,
    sex,
    COUNT(*) AS total_count,
    AVG(age) AS avg_age
  FROM {TABLE}
  GROUP BY race, sex
)
SELECT
  a.race,
  a.sex,
  a.total_count,
  ROUND(a.avg_age, 1) AS avg_age,
  t.workclass AS top_workclass
FROM agg a
LEFT JOIN top_workclass t ON a.race = t.race AND a.sex = t.sex
ORDER BY a.total_count DESC
LIMIT 100;"""
    },
    {
        "question": "Show education, sex, total people, percentage with income >50K, average hours, and most common occupation per group.",
        "sql": """WITH occ_counts AS (
  SELECT education, sex, occupation, COUNT(*) AS occ_cnt
  FROM {TABLE}
  GROUP BY education, sex, occupation
),
top_occ AS (
  SELECT education, sex, occupation
  FROM (
    SELECT education, sex, occupation, occ_cnt,
           ROW_NUMBER() OVER (PARTITION BY education, sex ORDER BY occ_cnt DESC, occupation ASC) AS rn
    FROM occ_counts
  )
  WHERE rn = 1
),
agg AS (
  SELECT
    education,
    sex,
    COUNT(*) AS total_people,
    AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income,
    AVG(hours_per_week) AS avg_hours
  FROM {TABLE}
  GROUP BY education, sex
)
SELECT
  a.education,
  a.sex,
  a.total_people,
  ROUND(a.pct_high_income * 100.0, 2) AS pct_high_income_percent,
  ROUND(a.avg_hours, 2) AS avg_hours,
  t.occupation AS top_occupation
FROM agg a
LEFT JOIN top_occ t ON a.education = t.education AND a.sex = t.sex
ORDER BY a.pct_high_income DESC, a.total_people DESC
LIMIT 100;"""
    }
]


def get_examples_for_table(table_name: str, simple: bool = True):
    """Get examples with table name replaced."""
    examples = SIMPLE_EXAMPLES_TEMPLATE if simple else COMPLEX_EXAMPLES_TEMPLATE
    result = []
    for ex in examples:
        result.append({
            "question": ex["question"],
            "sql": ex["sql"].replace("{TABLE}", table_name)
        })
    return result


def is_complex_query(question: str) -> bool:
    """Detect if a question requires complex SQL (CTEs, window functions, etc.)."""
    question_lower = question.lower()
    
    # Indicators of complexity
    complexity_keywords = [
        "most common", "most frequent", "top occupation", "top workclass",
        "most popular", "percentage", "percent", "ranked by",
        "for each", "per group", "show how many", "and the",
        "multiple aggregations", "combination of", "along with"
    ]
    
    # Multiple aggregations mentioned
    aggregation_count = sum(1 for word in ["count", "average", "sum", "percentage", "percent"] if word in question_lower)
    
    # Multiple metrics requested
    if aggregation_count >= 2:
        return True
    
    # Complexity keywords
    if any(keyword in question_lower for keyword in complexity_keywords):
        return True
    
    # Mentions finding "most common X per group"
    if ("most" in question_lower or "top" in question_lower) and ("per" in question_lower or "for each" in question_lower or "by" in question_lower):
        return True
    
    return False


def build_few_shot_prompt(schema: str, question: str) -> str:
    """Build Few-Shot prompt with simple examples."""
    # Original simple few-shot examples (hardcoded to adult_income)
    examples = [
        {
            "question": "Average hours_per_week by education (top 10).",
            "sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY education ORDER BY avg_hours DESC LIMIT 10;"
        },
        {
            "question": "Top 5 occupation by count where income = '>50K'.",
            "sql": "SELECT occupation, COUNT(*) AS count FROM adult_income WHERE income = '>50K' GROUP BY occupation ORDER BY count DESC LIMIT 5;"
        },
        {
            "question": "Count by race and sex.",
            "sql": "SELECT race, sex, COUNT(*) AS count FROM adult_income GROUP BY race, sex ORDER BY count DESC LIMIT 100;"
        },
        {
            "question": "How many people have sex = 'Female'?",
            "sql": "SELECT COUNT(*) AS count FROM adult_income WHERE sex = 'Female';"
        },
    ]
    
    prompt = f"""You are a SQL expert. Generate valid SQL queries using only the provided schema. Output a single SQL statement.

SCHEMA:
{schema}

Examples:

"""
    
    for ex in examples:
        prompt += f"Q: {ex['question']}\nSQL: {ex['sql']}\n\n"
    
    prompt += f"Q: {question}\nSQL:"
    return prompt


def build_cot_prompt(schema: str, question: str) -> str:
    """Build Chain-of-Thought prompt with explicit reasoning steps, enhanced for complex queries. Fully dynamic."""
    # Extract table name for reference (though not hardcoded in guidance)
    table_name = extract_table_name_from_schema(schema)
    
    is_complex = is_complex_query(question)
    
    if is_complex:
        prompt = f"""You are a SQL expert. Generate a valid SQL query using only the provided schema. This appears to be a COMPLEX query that may require CTEs (WITH clauses) and window functions.

SCHEMA:
{schema}

Question: {question}

Think step by step:

1. **Identify what needs to be calculated:**
   - What aggregations are needed? (COUNT, AVG, percentage, etc.)
   - Are there any "most common" or "top" items per group? (This requires ROW_NUMBER() window function)
   - What groups are needed? (GROUP BY columns)

2. **Plan the query structure:**
   - For complex queries with multiple steps, use CTEs (WITH clauses)
   - If finding "most common X per group", create a CTE with ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)
   - If calculating percentages, use AVG(CASE WHEN condition THEN 1.0 ELSE 0.0 END)
   - Aggregate statistics in a separate CTE
   - Join CTEs together in the final SELECT

3. **Write CTEs (if needed):**
   - CTE 1: Count occurrences (e.g., occupation counts per group)
   - CTE 2: Find top item using ROW_NUMBER() OVER (PARTITION BY group ORDER BY count DESC)
   - CTE 3: Aggregate statistics (COUNT, AVG, percentages)

4. **Final SELECT:**
   - Join CTEs together
   - Format output with ROUND() where appropriate
   - Add ORDER BY and LIMIT if specified

5. **Generate the complete SQL statement:**

SQL:"""
    else:
        prompt = f"""You are a SQL expert. Generate a valid SQL query using only the provided schema. Prefer LIMIT 100 if not specified.

SCHEMA:
{schema}

Question: {question}

Think step by step:

1. Identify relevant columns from the schema that match the question.
2. Determine any filters (WHERE clauses) needed.
3. Identify any aggregations (GROUP BY, COUNT, AVG, etc.) required.
4. Determine sorting (ORDER BY) and limits if specified.
5. Write the final SQL statement.

SQL:"""
    
    return prompt


def build_ltm_prompt(schema: str, question: str) -> str:
    """Build Least-to-Most prompt with substeps."""
    prompt = f"""You are a SQL expert. Generate a valid SQL query using only the provided schema. Prefer LIMIT 100 if not specified.

SCHEMA:
{schema}

Question: {question}

Break this down into substeps:

A. Identify the main table and columns needed.
B. Determine filtering conditions (if any).
C. Determine grouping and aggregation (if any).
D. Determine ordering and limits (if any).

Now produce the complete SQL:

SQL:"""
    
    return prompt


def build_eg_prompt(schema: str, question: str) -> str:
    """Build Execution-Guided prompt: direct SQL generation, enhanced for complex queries. Fully dynamic."""
    is_complex = is_complex_query(question)
    
    if is_complex:
        prompt = f"""You are a SQL expert. Generate a valid SQL query using only the provided schema. This is a COMPLEX query that requires CTEs and possibly window functions.

SCHEMA:
{schema}

Question: {question}

IMPORTANT GUIDELINES FOR COMPLEX QUERIES:
- Use WITH clauses (CTEs) to break down complex logic into steps
- To find "most common X per group", use: ROW_NUMBER() OVER (PARTITION BY group_cols ORDER BY count DESC)
- For percentages: AVG(CASE WHEN condition THEN 1.0 ELSE 0.0 END) * 100
- Format numbers with ROUND(value, decimals)
- Join CTEs in the final SELECT using LEFT JOIN on the grouping columns

Generate the complete SQL statement:

SQL:"""
    else:
        prompt = f"""You are a SQL expert. Generate a valid SQL query using only the provided schema. Prefer LIMIT 100 if not specified.

SCHEMA:
{schema}

Question: {question}

SQL:"""
    
    return prompt


def build_refine_prompt(
    schema: str,
    question: str,
    failed_sql: str,
    error_message: str
) -> str:
    """Build Execution-Guided refinement prompt, enhanced for complex queries. Fully dynamic."""
    is_complex = is_complex_query(question)
    
    if is_complex:
        prompt = f"""The following SQL query failed with an error. Generate a corrected SQL query using CTEs if needed.

SCHEMA:
{schema}

Question: {question}

Failed SQL:
{failed_sql}

Error:
{error_message}

CORRECTION GUIDELINES:
- If the query is complex, use WITH clauses (CTEs) to break it into steps
- For "most common X per group": Create a CTE with counts, then use ROW_NUMBER() OVER (PARTITION BY group ORDER BY count DESC) to find top item
- For percentages: Use AVG(CASE WHEN condition THEN 1.0 ELSE 0.0 END) * 100.0
- Join CTEs properly: LEFT JOIN on all grouping columns
- Use ROUND() for formatting numeric results
- Ensure all columns in SELECT are either aggregated or in GROUP BY

Corrected SQL:"""
    else:
        prompt = f"""The following SQL query failed with an error. Generate a corrected SQL query.

SCHEMA:
{schema}

Question: {question}

Failed SQL:
{failed_sql}

Error:
{error_message}

Corrected SQL:"""
    
    return prompt
