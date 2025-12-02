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



# Removed expand_schema_snippet import - schema is already a string from get_schema_snippet

def format_for_model(instruction, content, model_name):
    """
    Format the prompt based on the model.
    - Fine-tuned: Uses Alpaca format with specific instruction (matches training data).
    - Others: Concatenates instruction and content.
    """
    import os
    # Check if using fine-tuned Arctic model (port 11437 - uses Alpaca format from training)
    is_finetuned_arctic = (
        "arctic" in model_name.lower() and "finetuned" in model_name.lower() or
        os.getenv("OLLAMA_BASE_URL", "").endswith("11437")  # Fine-tuned Arctic server port
    )
    
    # Check if using fine-tuned model (either by model name or OLLAMA_BASE_URL)
    is_finetuned = (
        "finetuned" in model_name.lower() or 
        is_finetuned_arctic  # Fine-tuned Arctic
    )
    
    if is_finetuned or is_finetuned_arctic:
        # EXACT format from training data (prepare_spider_data.py line 70)
        # Input must end with "\nSQL:" and Response should be just SQL
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{content}
SQL:

### Response:
"""
    else:
        return f"{instruction}\n\n{content}\n\nSQL:"

def build_few_shot_prompt(schema, question, examples=None, model_name=""):
    """Build Few-Shot prompt."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Match training data instruction exactly
    instruction = "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question."
    
    content = f"SCHEMA:\n{schema_text}\n"
    if examples:
        formatted_examples = []
        for ex in examples:
            if isinstance(ex, dict):
                formatted_examples.append(f"Q: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}")
            else:
                formatted_examples.append(str(ex))
        content += "\nEXAMPLES:\n" + "\n\n".join(formatted_examples)
    
    content += f"\n\nQ: {question}"
    
    return format_for_model(instruction, content, model_name)

def build_cot_prompt(schema, question, model_name=""):
    """Build Chain-of-Thought prompt."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    instruction = "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question."
    
    # For fine-tuned: match training format (SCHEMA: ... Q: ...)
    # CoT reasoning can be in the instruction or after Q:
    content = f"""SCHEMA:
{schema_text}

Q: {question}"""

    return format_for_model(instruction, content, model_name)

def build_ltm_prompt(schema, question, model_name=""):
    """Build Least-to-Most prompt."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    instruction = "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question."
    
    # Match training format: SCHEMA: ... Q: ...
    content = f"""SCHEMA:
{schema_text}

Q: {question}"""

    return format_for_model(instruction, content, model_name)

def build_eg_prompt(schema, question, model_name=""):
    """Build Execution-Guided prompt."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    instruction = "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question."
    
    # Match training format: SCHEMA: ... Q: ...
    content = f"SCHEMA:\n{schema_text}\n\nQ: {question}"
    
    return format_for_model(instruction, content, model_name)

def build_refine_prompt(schema, question, previous_sql, error_msg, model_name=""):
    """Build Refinement prompt."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    instruction = "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question. The previous SQL query failed with an error. Fix the SQL query based on the error message."
    
    # Match training format: SCHEMA: ... Q: ...
    content = f"""SCHEMA:
{schema_text}

Q: {question}

Failed SQL: {previous_sql}
Error: {error_msg}"""

    return format_for_model(instruction, content, model_name)
