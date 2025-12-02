"""
Full evaluation script for fine-tuned Arctic Text-to-SQL model.
Runs 100 evaluation questions, sends them to your inference server,
executes SQL on MySQL, and logs detailed result JSON.
"""

import sys
import os
import time
import json
import traceback
import pymysql
import requests
from pathlib import Path

# ============================================
# FIX IMPORT PATH (so we can import prompts.py)
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts import (
    build_few_shot_prompt,
    get_examples_for_table,
)

# ============================================
# DATABASE CONFIG
# ============================================
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "test"     # ← CHANGE IF YOU SET A PASSWORD
MYSQL_DB = "adultdb"
TABLE_NAME = "adult_income"

# ============================================
# MODEL INFERENCE SERVER CONFIG
# ============================================
MODEL_ENDPOINT = "http://localhost:11437/v1/chat/completions"
MODEL_NAME = "arctic-finetuned"

# ============================================
# LOAD 100 TEST QUESTIONS
# ============================================
# Paste your list of 100 questions here:
# Format: { "question": "...", "gold_sql": "..." }
TEST_CASES = [
        {
            "question": "What is the total number of people in each workclass?",
            "gold_sql": "SELECT workclass, COUNT(*) AS total FROM adult_income GROUP BY workclass ORDER BY total DESC;"
        },
        {
            "question": "For each race, show the average age and average hours per week.",
            "gold_sql": "SELECT race, AVG(age) AS avg_age, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY race ORDER BY avg_age DESC;"
        },
        {
            "question": "Which education level has the highest percentage of people earning more than 50K?",
            "gold_sql": "SELECT education, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY education ORDER BY pct_high_income DESC LIMIT 1;"
        },
        {
            "question": "What is the average capital gain for people who work more than 50 hours per week?",
            "gold_sql": "SELECT AVG(capital_gain) AS avg_capital_gain FROM adult_income WHERE hours_per_week > 50;"
        },
        {
            "question": "For each marital status, show the count and percentage earning more than 50K.",
            "gold_sql": "SELECT marital_status, COUNT(*) AS total, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY marital_status ORDER BY pct_high_income DESC;"
        },
        {
            "question": "Which occupation has the highest average capital gain (ignore '?', require at least 20 people)?",
            "gold_sql": "SELECT occupation, AVG(capital_gain) AS avg_capital_gain FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 20 ORDER BY avg_capital_gain DESC LIMIT 1;"
        },
        {
            "question": "For each relationship type, what is the average age by sex?",
            "gold_sql": "SELECT relationship, sex, AVG(age) AS avg_age FROM adult_income GROUP BY relationship, sex ORDER BY relationship, sex;"
        },
        {
            "question": "What percentage of women earn more than 50K compared to men?",
            "gold_sql": "SELECT sex, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY sex ORDER BY sex;"
        },
        {
            "question": "For each native country with at least 50 people, show the average education_num.",
            "gold_sql": "SELECT native_country, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY native_country HAVING COUNT(*) >= 50 ORDER BY avg_education_num DESC;"
        },
        {
            "question": "Which workclass has the highest average hours worked per week?",
            "gold_sql": "SELECT workclass, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY workclass ORDER BY avg_hours DESC LIMIT 1;"
        },
        {
            "question": "For each education level, show the distribution by sex (count and percentage).",
            "gold_sql": "SELECT education, sex, COUNT(*) AS cnt, 100.0 * COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY education) AS pct FROM adult_income GROUP BY education, sex ORDER BY education, cnt DESC;"
        },
        {
            "question": "What is the average age of people earning more than 50K by occupation (top 10, ignore '?')?",
            "gold_sql": "SELECT occupation, AVG(age) AS avg_age FROM adult_income WHERE income = '>50K' AND occupation <> '?' GROUP BY occupation ORDER BY avg_age DESC LIMIT 10;"
        },
        {
            "question": "For each race and sex combination, what is the average hours worked per week?",
            "gold_sql": "SELECT race, sex, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY race, sex ORDER BY race, sex;"
        },
        {
            "question": "Which marital status has the highest percentage of people working more than 40 hours per week?",
            "gold_sql": "SELECT marital_status, 100.0 * AVG(CASE WHEN hours_per_week > 40 THEN 1.0 ELSE 0.0 END) AS pct_over_40hrs FROM adult_income GROUP BY marital_status ORDER BY pct_over_40hrs DESC LIMIT 1;"
        },
        {
            "question": "For each occupation, show the average capital loss (ignore '?', require at least 25 people).",
            "gold_sql": "SELECT occupation, AVG(capital_loss) AS avg_capital_loss FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 25 ORDER BY avg_capital_loss DESC;"
        },
        {
            "question": "What is the average education_num for people earning <=50K vs >50K?",
            "gold_sql": "SELECT income, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY income ORDER BY income;"
        },
        {
            "question": "For each native country, show the top occupation by count (ignore '?', require at least 30 people in country).",
            "gold_sql": "WITH occ AS (SELECT native_country, occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY native_country, occupation), ranked AS (SELECT native_country, occupation, cnt, ROW_NUMBER() OVER (PARTITION BY native_country ORDER BY cnt DESC) AS rn FROM occ) SELECT r.native_country, r.occupation AS top_occupation, r.cnt FROM ranked r JOIN (SELECT native_country, COUNT(*) AS total FROM adult_income GROUP BY native_country HAVING total >= 30) c ON r.native_country = c.native_country WHERE r.rn = 1 ORDER BY r.cnt DESC;"
        },
        {
            "question": "Which relationship type has the highest average capital gain?",
            "gold_sql": "SELECT relationship, AVG(capital_gain) AS avg_capital_gain FROM adult_income GROUP BY relationship ORDER BY avg_capital_gain DESC LIMIT 1;"
        },
        {
            "question": "For each workclass and sex, what is the average hours worked per week?",
            "gold_sql": "SELECT workclass, sex, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY workclass, sex ORDER BY workclass, sex;"
        },
        {
            "question": "What is the average age by income class and sex?",
            "gold_sql": "SELECT income, sex, AVG(age) AS avg_age FROM adult_income GROUP BY income, sex ORDER BY income, sex;"
        },
        {
            "question": "For each education level, show the percentage of people in each income bracket.",
            "gold_sql": "SELECT education, income, 100.0 * COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY education) AS pct FROM adult_income GROUP BY education, income ORDER BY education, income;"
        },
        {
            "question": "Which occupations have the highest percentage of people working exactly 40 hours per week (ignore '?', require at least 20 people)?",
            "gold_sql": "SELECT occupation, 100.0 * AVG(CASE WHEN hours_per_week = 40 THEN 1.0 ELSE 0.0 END) AS pct_40hrs, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING cnt >= 20 ORDER BY pct_40hrs DESC LIMIT 10;"
        },
        {
            "question": "For each race, what is the most common workclass and its count?",
            "gold_sql": "WITH wc AS (SELECT race, workclass, COUNT(*) AS cnt FROM adult_income GROUP BY race, workclass), ranked AS (SELECT race, workclass, cnt, ROW_NUMBER() OVER (PARTITION BY race ORDER BY cnt DESC, workclass ASC) AS rn FROM wc) SELECT race, workclass AS top_workclass, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
        },
        {
            "question": "What is the average hours per week for people with capital_gain > 0 vs capital_gain = 0?",
            "gold_sql": "SELECT CASE WHEN capital_gain > 0 THEN 'Has Gain' ELSE 'No Gain' END AS gain_status, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY gain_status;"
        },
        {
            "question": "For each sex, show the top 5 education levels by count.",
            "gold_sql": "WITH edu AS (SELECT sex, education, COUNT(*) AS cnt FROM adult_income GROUP BY sex, education), ranked AS (SELECT sex, education, cnt, ROW_NUMBER() OVER (PARTITION BY sex ORDER BY cnt DESC, education ASC) AS rn FROM edu) SELECT sex, education, cnt FROM ranked WHERE rn <= 5 ORDER BY sex, cnt DESC;"
        },
        {
            "question": "Which native country has the highest average education_num (require at least 40 people)?",
            "gold_sql": "SELECT native_country, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY native_country HAVING COUNT(*) >= 40 ORDER BY avg_education_num DESC LIMIT 1;"
        },
        {
            "question": "For each occupation, show the average age and average hours per week (ignore '?', top 15 by average age).",
            "gold_sql": "SELECT occupation, AVG(age) AS avg_age, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE occupation <> '?' GROUP BY occupation ORDER BY avg_age DESC LIMIT 15;"
        },
        {
            "question": "What is the distribution of people by marital status and income?",
            "gold_sql": "SELECT marital_status, income, COUNT(*) AS cnt FROM adult_income GROUP BY marital_status, income ORDER BY marital_status, income;"
        },
        {
            "question": "For each relationship type, what percentage of people earn more than 50K?",
            "gold_sql": "SELECT relationship, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY relationship ORDER BY pct_high_income DESC;"
        },
        {
            "question": "Which workclass has the highest percentage of people with capital_gain > 0?",
            "gold_sql": "SELECT workclass, 100.0 * AVG(CASE WHEN capital_gain > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_gain FROM adult_income GROUP BY workclass ORDER BY pct_with_gain DESC LIMIT 1;"
        },
        {
            "question": "For each education level, show the average capital gain and capital loss.",
            "gold_sql": "SELECT education, AVG(capital_gain) AS avg_capital_gain, AVG(capital_loss) AS avg_capital_loss FROM adult_income GROUP BY education ORDER BY avg_capital_gain DESC;"
        },
        {
            "question": "What is the average hours per week by age group (under 30, 30-50, over 50)?",
            "gold_sql": "SELECT CASE WHEN age < 30 THEN 'Under 30' WHEN age <= 50 THEN '30-50' ELSE 'Over 50' END AS age_group, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY age_group ORDER BY age_group;"
        },
        {
            "question": "For each race, show the most common education level and its count.",
            "gold_sql": "WITH edu AS (SELECT race, education, COUNT(*) AS cnt FROM adult_income GROUP BY race, education), ranked AS (SELECT race, education, cnt, ROW_NUMBER() OVER (PARTITION BY race ORDER BY cnt DESC, education ASC) AS rn FROM edu) SELECT race, education AS top_education, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
        },
        {
            "question": "Which occupation has the highest average hours per week among people earning more than 50K (ignore '?', require at least 15 people)?",
            "gold_sql": "SELECT occupation, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE income = '>50K' AND occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 15 ORDER BY avg_hours DESC LIMIT 1;"
        },
        {
            "question": "For each sex and marital status, what is the average education_num?",
            "gold_sql": "SELECT sex, marital_status, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY sex, marital_status ORDER BY sex, avg_education_num DESC;"
        },
        {
            "question": "What is the percentage of people working more than 45 hours per week by income class?",
            "gold_sql": "SELECT income, 100.0 * AVG(CASE WHEN hours_per_week > 45 THEN 1.0 ELSE 0.0 END) AS pct_over_45hrs FROM adult_income GROUP BY income ORDER BY income;"
        },
        {
            "question": "For each native country, show the average age by income class (require at least 25 people per country).",
            "gold_sql": "SELECT native_country, income, AVG(age) AS avg_age FROM adult_income GROUP BY native_country, income HAVING COUNT(*) >= 25 ORDER BY native_country, income;"
        },
        {
            "question": "Which workclass has the highest average age?",
            "gold_sql": "SELECT workclass, AVG(age) AS avg_age FROM adult_income GROUP BY workclass ORDER BY avg_age DESC LIMIT 1;"
        },
        {
            "question": "For each relationship type, show the count and average hours per week.",
            "gold_sql": "SELECT relationship, COUNT(*) AS cnt, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY relationship ORDER BY cnt DESC;"
        },
        {
            "question": "What is the average capital gain for people with different education levels (top 10 by average gain)?",
            "gold_sql": "SELECT education, AVG(capital_gain) AS avg_capital_gain FROM adult_income GROUP BY education ORDER BY avg_capital_gain DESC LIMIT 10;"
        },
        {
            "question": "For each sex, what is the most common occupation (ignore '?')?",
            "gold_sql": "WITH occ AS (SELECT sex, occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY sex, occupation), ranked AS (SELECT sex, occupation, cnt, ROW_NUMBER() OVER (PARTITION BY sex ORDER BY cnt DESC, occupation ASC) AS rn FROM occ) SELECT sex, occupation AS top_occupation, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
        },
        {
            "question": "Which education level has the highest average hours per week among people earning more than 50K?",
            "gold_sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE income = '>50K' GROUP BY education ORDER BY avg_hours DESC LIMIT 1;"
        },
        {
            "question": "For each race, show the percentage of people in each income bracket.",
            "gold_sql": "SELECT race, income, 100.0 * COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY race) AS pct FROM adult_income GROUP BY race, income ORDER BY race, income;"
        },
        {
            "question": "What is the average age by workclass and sex?",
            "gold_sql": "SELECT workclass, sex, AVG(age) AS avg_age FROM adult_income GROUP BY workclass, sex ORDER BY workclass, sex;"
        },
        {
            "question": "For each occupation, show the average education_num (ignore '?', top 10 by average education).",
            "gold_sql": "SELECT occupation, AVG(education_num) AS avg_education_num FROM adult_income WHERE occupation <> '?' GROUP BY occupation ORDER BY avg_education_num DESC LIMIT 10;"
        },
        {
            "question": "Which marital status has the highest percentage of people with capital_loss > 0?",
            "gold_sql": "SELECT marital_status, 100.0 * AVG(CASE WHEN capital_loss > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_loss FROM adult_income GROUP BY marital_status ORDER BY pct_with_loss DESC LIMIT 1;"
        },
        {
            "question": "For each native country, show the average hours per week (require at least 30 people, top 15).",
            "gold_sql": "SELECT native_country, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY native_country HAVING COUNT(*) >= 30 ORDER BY avg_hours DESC LIMIT 15;"
        },
        {
            "question": "What is the distribution of people by relationship type and income?",
            "gold_sql": "SELECT relationship, income, COUNT(*) AS cnt FROM adult_income GROUP BY relationship, income ORDER BY relationship, income;"
        },
        {
            "question": "For each education level, show the most common workclass and its count.",
            "gold_sql": "WITH wc AS (SELECT education, workclass, COUNT(*) AS cnt FROM adult_income GROUP BY education, workclass), ranked AS (SELECT education, workclass, cnt, ROW_NUMBER() OVER (PARTITION BY education ORDER BY cnt DESC, workclass ASC) AS rn FROM wc) SELECT education, workclass AS top_workclass, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
        },
        {
            "question": "Which race has the highest average capital gain?",
            "gold_sql": "SELECT race, AVG(capital_gain) AS avg_capital_gain FROM adult_income GROUP BY race ORDER BY avg_capital_gain DESC LIMIT 1;"
        },
        {
            "question": "For each sex, what is the average age and average education_num?",
            "gold_sql": "SELECT sex, AVG(age) AS avg_age, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY sex ORDER BY sex;"
        },
        {
            "question": "What is the percentage of people working less than 30 hours per week by income class?",
            "gold_sql": "SELECT income, 100.0 * AVG(CASE WHEN hours_per_week < 30 THEN 1.0 ELSE 0.0 END) AS pct_under_30hrs FROM adult_income GROUP BY income ORDER BY income;"
        },
        {
            "question": "For each occupation, show the average age (ignore '?', require at least 20 people, top 10).",
            "gold_sql": "SELECT occupation, AVG(age) AS avg_age FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 20 ORDER BY avg_age DESC LIMIT 10;"
        }
    ]

NUM_TEST_QUESTIONS = len(TEST_CASES)

# ============================================
# MYSQL CONNECTION
# ============================================
def get_mysql_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )

# ============================================
# GET SCHEMA FOR PROMPTING
# ============================================
def get_schema_snippet():
    conn = get_mysql_conn()
    with conn.cursor() as cur:
        cur.execute(f"DESCRIBE {TABLE_NAME}")
        rows = cur.fetchall()

    schema = f"{TABLE_NAME}(" + ", ".join([row["Field"] for row in rows]) + ")"
    return schema

# ============================================
# SEND PROMPT TO MODEL SERVER
# ============================================
def call_model(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    response = requests.post(MODEL_ENDPOINT, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Model server error: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()

# ============================================
# EXECUTE SQL SAFELY
# ============================================
def run_sql(sql: str):
    sql_clean = sql.strip().rstrip(";") + ";"  # ensure one semicolon

    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_clean)
            rows = cur.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        conn.close()
        return False, None, str(e)

# ============================================
# MAIN EVALUATION LOOP
# ============================================
def evaluate():
    print("Fetching schema from DB...")
    schema_snippet = get_schema_snippet()
    print("Schema:", schema_snippet)

    results = []

    print(f"\nRunning evaluation on {NUM_TEST_QUESTIONS} questions...\n")

    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        gold_sql = case["gold_sql"]

        print(f"=== [{i}/{NUM_TEST_QUESTIONS}] Question ===")
        print(question)

        # Few-shot examples (first 3 cases)
        few_shot_examples = [
            {"question": t["question"], "sql": t["gold_sql"]}
            for t in TEST_CASES[:3]
        ]

        prompt = build_few_shot_prompt(
            schema=schema_snippet,
            question=question,
            examples=few_shot_examples,
            model_name=MODEL_NAME,
        )

        try:
            model_sql = call_model(prompt)
            print("MODEL SQL:", model_sql)
        except Exception as e:
            print("Model failed:", e)
            results.append({
                "question": question,
                "model_sql": None,
                "gold_sql": gold_sql,
                "success": False,
                "error": str(e),
                "rows": None,
            })
            continue

        # Execute model SQL
        success, rows, error = run_sql(model_sql)

        results.append({
            "question": question,
            "model_sql": model_sql,
            "gold_sql": gold_sql,
            "success": success,
            "error": error,
            "rows": len(rows) if rows else 0,
        })

        print("Success:", success)
        print("Rows:", len(rows) if rows else 0)
        if error:
            print("Error:", error)
        print("-------------------------------------------\n")

    # SAVE RESULTS
    out_path = SCRIPT_DIR / "adult_model_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n====================================")
    print("   Evaluation complete!")
    print(f"   Results saved to: {out_path}")
    print("====================================")

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    evaluate()
