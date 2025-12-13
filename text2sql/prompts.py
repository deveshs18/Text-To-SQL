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
    Format the prompt based on model type:
    - Base models (not fine-tuned): Use simpler, more instructional format
    - GPT: Use instruction-based format (original GPT format)
    - Arctic: Use simple SCHEMA: ... Q: ... SQL: (no hints, no extra guidance)
    - Qwen Finetuned: Use exact training format SCHEMA: ... Q: ... SQL: (no hints, no extra guidance)
    """
    import os
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
    is_arctic = "arctic" in model_name.lower()
    is_qwen_finetuned = (
        "qwen-0.5b-spider" in model_name.lower() or
        ("qwen" in model_name.lower() and "spider" in model_name.lower())
    )
    
    # For base models: Use simpler, instructional format
    if is_base_model:
        # Extract schema and question
        schema_match = re.search(r'SCHEMA:?\s*(.*?)(?=\n\s*(?:Q:|EXAMPLES:)|$)', content, re.DOTALL)
        if schema_match:
            schema_text = schema_match.group(1).strip()
            # Remove hints (the | Note: parts) for base model
            schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
            schema_text = ' '.join(schema_text.split())
        else:
            if 'SCHEMA:' in content:
                schema_text = content.split('SCHEMA:')[1].split('\n')[0].strip()
                schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE)
            else:
                schema_text = ""
        
        # Extract question
        all_q_matches = list(re.finditer(r'Q:\s*(.*?)(?=\n\s*(?:SQL:|Q:|$))', content, re.DOTALL))
        if all_q_matches:
            question_text = all_q_matches[-1].group(1).strip()
        else:
            if 'Q:' in content:
                q_parts = content.split('Q:')
                if len(q_parts) > 1:
                    last_q_part = q_parts[-1]
                    question_text = last_q_part.split('SQL:')[0].strip() if 'SQL:' in last_q_part else last_q_part.strip()
                else:
                    question_text = ""
            else:
                question_text = ""
        
        # Build simpler format for base model (more instructional)
        # Remove any thinking guidance for base model - it confuses it
        question_clean = re.sub(r'Think step by step:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
        question_clean = re.sub(r'Break down.*$', '', question_clean, flags=re.IGNORECASE | re.DOTALL).strip()
        question_clean = re.sub(r'Generate SQL and verify.*$', '', question_clean, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Simple format: instruction + schema + question
        formatted_content = f"""You are a SQL expert. Generate a valid SQL query for the given question.

Database schema:
{schema_text}

Question: {question_clean}

SQL query:"""
        
        return formatted_content
    
    # For GPT: Use instruction-based format (original GPT format)
    if is_gpt:
        # Extract schema
        schema_match = re.search(r'SCHEMA:?\s*(.*?)(?=\n\s*(?:Q:|EXAMPLES:)|$)', content, re.DOTALL)
        if schema_match:
            schema_text = schema_match.group(1).strip()
            # Remove hints for GPT (they might confuse it)
            schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
            schema_text = ' '.join(schema_text.split())
        else:
            if 'SCHEMA:' in content:
                schema_text = content.split('SCHEMA:')[1].split('\n')[0].strip()
                schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE)
            else:
                schema_text = ""
        
        # Check if examples are present (Few-Shot) - look for multiple Q: ... SQL: patterns
        # Split content by Q: to find all question-SQL pairs
        q_parts = content.split('Q:')
        if len(q_parts) > 2:  # More than just schema and one question (has examples)
            # Examples are present - extract them
            examples_text = ""
            # Process each Q: ... SQL: pair (skip first part which is SCHEMA:)
            for i, part in enumerate(q_parts[1:-1], 1):  # Skip first (schema) and last (actual question)
                # Split by SQL: to get question and SQL
                if 'SQL:' in part:
                    q_text, sql_part = part.split('SQL:', 1)
                    q_text = q_text.strip()
                    sql_text = sql_part.split('Q:')[0].strip() if 'Q:' in sql_part else sql_part.strip()
                    # Remove any trailing Q: markers
                    sql_text = sql_text.split('\nQ:')[0].strip()
                    examples_text += f"\nExample {i}:\nQuestion: {q_text}\nSQL: {sql_text}"
            
            # Get the actual question (last Q:)
            last_part = q_parts[-1]
            question_text = last_part.split('SQL:')[0].strip() if 'SQL:' in last_part else last_part.strip()
            question_text = re.sub(r'Think step by step:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            question_text = re.sub(r'Break down.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            question_text = re.sub(r'Generate SQL and verify.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            
            # GPT format with examples
            formatted_content = f"""You are a SQL expert. Generate a valid SQL query for the given question.

Database schema:
{schema_text}
{examples_text}

Question: {question_text}

SQL query:"""
        else:
            # No examples - simple format
            # Extract question from content
            if 'Q:' in content:
                q_parts = content.split('Q:')
                if len(q_parts) > 1:
                    last_q_part = q_parts[-1]
                    question_text = last_q_part.split('SQL:')[0].strip() if 'SQL:' in last_q_part else last_q_part.strip()
                else:
                    question_text = ""
            else:
                question_text = ""
            
            # Remove extra guidance text for GPT
            question_text = re.sub(r'Think step by step:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            question_text = re.sub(r'Break down.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            question_text = re.sub(r'Generate SQL and verify.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            
            # GPT format: Instruction + schema + question
            formatted_content = f"""You are a SQL expert. Generate a valid SQL query for the given question.

Database schema:
{schema_text}

Question: {question_text}

SQL query:"""
        return formatted_content
    
    # For Arctic and Qwen Finetuned: Use simple SCHEMA: ... Q: ... SQL: format
    # Extract schema part (everything after SCHEMA: until Q: or EXAMPLES:)
    schema_match = re.search(r'SCHEMA:?\s*(.*?)(?=\n\s*(?:Q:|EXAMPLES:)|$)', content, re.DOTALL)
    if schema_match:
        schema_text = schema_match.group(1).strip()
        # Remove hints for Arctic and Qwen (they weren't trained with hints)
        schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove newlines and extra whitespace, make single line
        schema_text = ' '.join(schema_text.split())
    else:
        # Fallback: try to extract from content
        if 'SCHEMA:' in content:
            schema_text = content.split('SCHEMA:')[1].split('\n')[0].strip()
            schema_text = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE)
        else:
            schema_text = ""
    
    # Extract question (Q: ...)
    # IMPORTANT: If examples are present, extract the LAST Q: (the actual question)
    all_q_matches = list(re.finditer(r'Q:\s*(.*?)(?=\n\s*(?:SQL:|Q:|$))', content, re.DOTALL))
    if all_q_matches:
        # Get the LAST question (the actual one, after examples)
        question_match = all_q_matches[-1]
        question_text = question_match.group(1).strip()
        # Remove extra guidance text (Arctic and Qwen weren't trained with this)
        question_text = re.sub(r'Think step by step:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
        question_text = re.sub(r'Break down.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
        question_text = re.sub(r'Generate SQL and verify.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        # Fallback
        if 'Q:' in content:
            q_parts = content.split('Q:')
            if len(q_parts) > 1:
                last_q_part = q_parts[-1]
                question_text = last_q_part.split('SQL:')[0].strip() if 'SQL:' in last_q_part else last_q_part.strip()
                question_text = re.sub(r'Think step by step:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
                question_text = re.sub(r'Break down.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
                question_text = re.sub(r'Generate SQL and verify.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                question_text = ""
        else:
            question_text = ""
    
    # Build simple format: SCHEMA: ... Q: ... SQL: (no hints, no extra guidance)
    if '\nQ:' in content and content.count('Q:') > 1:
        # Examples are present (for Few-Shot)
        # The content already has the format: SCHEMA: ... Q: example1 SQL: sql1 Q: actual_question
        # We just need to ensure it ends with SQL: and remove any hints
        formatted_content = content.strip()
        # Remove hints
        formatted_content = re.sub(r'\s*\|\s*Note:.*$', '', formatted_content, flags=re.IGNORECASE | re.MULTILINE)
        # Ensure it ends with SQL: (not just Q:)
        if not formatted_content.rstrip().endswith('SQL:'):
            # If it ends with just Q: question_text, add SQL:
            if formatted_content.rstrip().endswith(question_text):
                formatted_content = formatted_content.rstrip() + '\nSQL:'
            else:
                formatted_content = formatted_content.rstrip() + '\nSQL:'
    else:
        # No examples, simple format
        formatted_content = f"SCHEMA: {schema_text}\nQ: {question_text}\nSQL:"
    
    return formatted_content
    
    # OLD CODE (kept for reference, but now all models use standardized format above)
    if False:  # This block never executes, kept for reference
        is_qwen = "qwen" in model_name.lower()
        # Qwen was trained on exact format: SCHEMA: table1(col1, col2) | table2(col3, col4)
        # Q: question
        # SQL: sql
        # Remove any instruction text, remove EXAMPLES: header, ensure single-line SCHEMA
        
        # Extract schema part (everything after SCHEMA: until Q: or EXAMPLES:)
        schema_match = re.search(r'SCHEMA:?\s*(.*?)(?=\n\s*(?:Q:|EXAMPLES:)|$)', content, re.DOTALL)
        if schema_match:
            schema_text = schema_match.group(1).strip()
            # Remove newlines and extra whitespace, make single line
            schema_text = ' '.join(schema_text.split())
        else:
            # Fallback: try to extract from content
            if 'SCHEMA:' in content:
                schema_text = content.split('SCHEMA:')[1].split('\n')[0].strip()
            else:
                schema_text = ""
        
        # Extract question (Q: ...)
        # IMPORTANT: If examples are present, extract the LAST Q: (the actual question)
        # If no examples, extract the first Q:
        all_q_matches = list(re.finditer(r'Q:\s*(.*?)(?=\n\s*(?:SQL:|Q:|$))', content, re.DOTALL))
        if all_q_matches:
            # Get the LAST question (the actual one, after examples)
            question_match = all_q_matches[-1]
            question_text = question_match.group(1).strip()
        else:
            # Fallback
            if 'Q:' in content:
                # Split by Q: and take the last one (actual question)
                q_parts = content.split('Q:')
                if len(q_parts) > 1:
                    # Get the last Q: part, then extract until SQL: or end
                    last_q_part = q_parts[-1]
                    question_text = last_q_part.split('SQL:')[0].strip() if 'SQL:' in last_q_part else last_q_part.strip()
                else:
                    question_text = ""
            else:
                question_text = ""
        
        # This code is now replaced by standardized format above
        pass

def build_few_shot_prompt(schema, question, examples=None, model_name=""):
    """Build Few-Shot prompt - Model-specific format."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    # Detect model type
    is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
    
    if is_base_model:
        # Base model: Simpler format, no hints, no examples (they confuse it)
        # Make schema single-line, remove hints
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        # format_for_model will handle base model formatting
        return format_for_model("", content, model_name)
    elif is_gpt:
        # GPT: No hints, but can use examples
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}"
        
        # Add examples if provided (for Few-Shot)
        if examples:
            for ex in examples:
                if isinstance(ex, dict):
                    content += f"\nQ: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}"
                else:
                    content += f"\n{ex}"
        
        # Add the actual question
        content += f"\nQ: {question}"
        
        # format_for_model will convert to GPT format
        return format_for_model("", content, model_name)
    else:
        # Arctic and Qwen Finetuned: No hints, but can use examples
        # Make schema single-line, remove hints
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}"
        
        # Add examples if provided (for Few-Shot)
        if examples:
            for ex in examples:
                if isinstance(ex, dict):
                    content += f"\nQ: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}"
                else:
                    content += f"\n{ex}"
        
        # Add the actual question
        content += f"\nQ: {question}"
        
        # Use simple format (SCHEMA: ... Q: ... SQL:)
        return format_for_model("", content, model_name)

def build_cot_prompt(schema, question, model_name=""):
    """Build Chain-of-Thought prompt - Model-specific format."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    # Detect model type
    is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
    
    if is_base_model:
        # Base model: Simpler CoT, no hints
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        # format_for_model will add simpler instruction for base model
        return format_for_model("", content, model_name)
    elif is_gpt:
        # GPT: Simple CoT (no extra guidance text, GPT handles it naturally)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)
    else:
        # Arctic and Qwen Finetuned: Simple format (no extra guidance - they weren't trained with it)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)

def build_ltm_prompt(schema, question, model_name=""):
    """Build Least-to-Most prompt - Model-specific format."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    # Detect model type
    is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
    
    if is_base_model:
        # Base model: Simpler LtM, no hints
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        # format_for_model will add simpler instruction for base model
        return format_for_model("", content, model_name)
    elif is_gpt:
        # GPT: Simple LtM (no extra guidance text, GPT handles it naturally)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)
    else:
        # Arctic and Qwen Finetuned: Simple format (no extra guidance - they weren't trained with it)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)

def build_eg_prompt(schema, question, model_name=""):
    """Build Execution-Guided prompt - Model-specific format."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    # Detect model type
    is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
    
    if is_base_model:
        # Base model: Simpler EG, no hints
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        # format_for_model will add simpler instruction for base model
        return format_for_model("", content, model_name)
    elif is_gpt:
        # GPT: Simple EG (no extra guidance text, GPT handles it naturally)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)
    else:
        # Arctic and Qwen Finetuned: Simple format (no extra guidance - they weren't trained with it)
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}"
        return format_for_model("", content, model_name)

def build_refine_prompt(schema, question, previous_sql, error_msg, model_name=""):
    """Build Refinement prompt - Model-specific format."""
    # schema is already a string from get_schema_snippet, use it directly
    schema_text = schema if isinstance(schema, str) else str(schema)
    
    # Detect model type
    is_base_model = (
        "qwen-0.5b-base" in model_name.lower() or
        ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
    )
    
    if is_base_model:
        # Base model: Simpler refinement format
        schema_clean = re.sub(r'\s*\|\s*Note:.*$', '', schema_text, flags=re.IGNORECASE | re.MULTILINE)
        schema_single_line = ' '.join(schema_clean.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}\nFailed SQL: {previous_sql}\nError: {error_msg}"
        return format_for_model("Fix the SQL query based on the error message.", content, model_name)
    else:
        # Finetuned models: Standardized format
        schema_single_line = ' '.join(schema_text.split())
        content = f"SCHEMA: {schema_single_line}\nQ: {question}\nFailed SQL: {previous_sql}\nError: {error_msg}\nFix the SQL query based on the error message."
        return format_for_model("", content, model_name)
