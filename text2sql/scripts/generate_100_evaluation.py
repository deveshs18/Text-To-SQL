"""
Comprehensive evaluation script for 100 test cases across both models (Ollama & GPT-4o-mini).
Generates Figure 1, Table 2, and Figure 6 as requested.
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Import compare_rag functions - need to set up environment first
os.chdir(parent_dir)  # Change to parent directory for imports
import importlib.util
compare_rag_path = script_dir / "compare_rag.py"
spec = importlib.util.spec_from_file_location("compare_rag", compare_rag_path)
compare_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_rag)
compare_rag_performance = compare_rag.compare_rag_performance
load_test_cases = compare_rag.load_test_cases

load_dotenv()

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Color scheme
COLORS = {
    'with_rag': '#2E86AB',  # Blue
    'without_rag': '#E63946',  # Red
    'improvement': '#06A77D'  # Green
}


def generate_50_additional_test_cases() -> List[Dict[str, str]]:
    """Generate 50 additional diverse test cases."""
    return [
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


def load_all_100_test_cases() -> List[Dict[str, str]]:
    """Load existing 50 test cases and add 50 more."""
    # Load existing test cases from compare_rag.py
    existing_cases = load_test_cases()
    
    # If we have existing cases, use them; otherwise start fresh
    if len(existing_cases) >= 50:
        existing_50 = existing_cases[:50]
    else:
        # Load from template if available
        template_path = parent_dir / "data" / "test_cases_template.json"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                existing_50 = json.load(f)
        else:
            existing_50 = []
    
    # Generate 50 additional cases
    additional_50 = generate_50_additional_test_cases()
    
    # Combine to get 100 total
    all_100 = existing_50 + additional_50
    
    print(f"Loaded {len(existing_50)} existing test cases")
    print(f"Generated {len(additional_50)} additional test cases")
    print(f"Total: {len(all_100)} test cases")
    
    return all_100


def run_evaluation_for_model(
    model_name: str,
    test_cases: List[Dict[str, str]],
    techniques: List[str],
    db_url: str,
    output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """Run evaluation for a single model across all techniques."""
    print(f"\n{'='*80}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for technique in techniques:
        print(f"\n--- Technique: {technique} ---")
        df = compare_rag_performance(
            test_cases,
            model_name,
            technique,
            db_url,
            max_workers=10 if 'openai' in model_name else 4
        )
        results[technique] = df
        
        # Save individual results
        output_file = output_dir / f"{model_name.replace('/', '_')}_{technique}_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")
    
    return results


def calculate_overall_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall metrics from results dataframe."""
    total = len(df)
    
    # Success rate
    with_rag_success = sum(df['With_RAG_Success'] == '✅')
    without_rag_success = sum(df['Without_RAG_Success'] == '✅')
    
    # Execution Accuracy
    with_rag_ex = sum(df['With_RAG_EX'] == '✅')
    without_rag_ex = sum(df['Without_RAG_EX'] == '✅')
    
    # Semantic Match
    with_rag_sm = sum(df['With_RAG_SM'] == '✅')
    without_rag_sm = sum(df['Without_RAG_SM'] == '✅')
    
    # F1-Score (convert to float)
    with_rag_f1 = pd.to_numeric(df['With_RAG_F1'], errors='coerce').mean()
    without_rag_f1 = pd.to_numeric(df['Without_RAG_F1'], errors='coerce').mean()
    
    # Latency (extract numeric values)
    with_rag_latency = df['With_RAG_Latency'].str.replace('s', '').astype(float).mean()
    without_rag_latency = df['Without_RAG_Latency'].str.replace('s', '').astype(float).mean()
    
    return {
        'total': total,
        'with_rag_success_pct': (with_rag_success / total) * 100,
        'without_rag_success_pct': (without_rag_success / total) * 100,
        'with_rag_ex_pct': (with_rag_ex / total) * 100,
        'without_rag_ex_pct': (without_rag_ex / total) * 100,
        'with_rag_sm_pct': (with_rag_sm / total) * 100,
        'without_rag_sm_pct': (without_rag_sm / total) * 100,
        'with_rag_f1': with_rag_f1,
        'without_rag_f1': without_rag_f1,
        'with_rag_latency': with_rag_latency,
        'without_rag_latency': without_rag_latency,
        'with_rag_success_count': with_rag_success,
        'without_rag_success_count': without_rag_success,
        'with_rag_ex_count': with_rag_ex,
        'without_rag_ex_count': without_rag_ex
    }


def create_figure_1(metrics_ollama: Dict, metrics_gpt: Dict, output_dir: Path):
    """Create Figure 1: Overall metrics comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics = ['Success Rate', 'EX', 'SM', 'F1-Score']
    x = range(len(metrics))
    width = 0.35
    
    # Ollama subplot
    ollama_with = [
        metrics_ollama['with_rag_success_pct'],
        metrics_ollama['with_rag_ex_pct'],
        metrics_ollama['with_rag_sm_pct'],
        metrics_ollama['with_rag_f1'] * 100
    ]
    ollama_without = [
        metrics_ollama['without_rag_success_pct'],
        metrics_ollama['without_rag_ex_pct'],
        metrics_ollama['without_rag_sm_pct'],
        metrics_ollama['without_rag_f1'] * 100
    ]
    
    ax1.bar([i - width/2 for i in x], ollama_with, width, label='With RAG', color=COLORS['with_rag'])
    ax1.bar([i + width/2 for i in x], ollama_without, width, label='Without RAG', color=COLORS['without_rag'])
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Percentage / Score', fontweight='bold')
    ax1.set_title('Ollama (llama3.1:8b): Overall Performance', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # GPT subplot
    gpt_with = [
        metrics_gpt['with_rag_success_pct'],
        metrics_gpt['with_rag_ex_pct'],
        metrics_gpt['with_rag_sm_pct'],
        metrics_gpt['with_rag_f1'] * 100
    ]
    gpt_without = [
        metrics_gpt['without_rag_success_pct'],
        metrics_gpt['without_rag_ex_pct'],
        metrics_gpt['without_rag_sm_pct'],
        metrics_gpt['without_rag_f1'] * 100
    ]
    
    ax2.bar([i - width/2 for i in x], gpt_with, width, label='With RAG', color=COLORS['with_rag'])
    ax2.bar([i + width/2 for i in x], gpt_without, width, label='Without RAG', color=COLORS['without_rag'])
    ax2.set_xlabel('Metrics', fontweight='bold')
    ax2.set_ylabel('Percentage / Score', fontweight='bold')
    ax2.set_title('GPT-4o-mini: Overall Performance', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = output_dir / "figure_1_overall_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 1 saved: {output_file}")
    plt.close()


def create_table_2(results_ollama: Dict[str, pd.DataFrame], results_gpt: Dict[str, pd.DataFrame], output_dir: Path):
    """Create Table 2: Technique-by-technique comparison."""
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    table_data = []
    
    for technique in techniques:
        # Ollama metrics
        ollama_df = results_ollama[technique]
        ollama_metrics = calculate_overall_metrics(ollama_df)
        
        # GPT metrics
        gpt_df = results_gpt[technique]
        gpt_metrics = calculate_overall_metrics(gpt_df)
        
        # Add rows for Ollama
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'EX',
            'With RAG': f"{ollama_metrics['with_rag_ex_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_ex_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_ex_pct'] - ollama_metrics['without_rag_ex_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'SM',
            'With RAG': f"{ollama_metrics['with_rag_sm_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_sm_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_sm_pct'] - ollama_metrics['without_rag_sm_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'F1',
            'With RAG': f"{ollama_metrics['with_rag_f1']:.3f}",
            'Without RAG': f"{ollama_metrics['without_rag_f1']:.3f}",
            'Improvement': f"+{ollama_metrics['with_rag_f1'] - ollama_metrics['without_rag_f1']:.3f}"
        })
        
        # Add rows for GPT
        table_data.append({
            'Model': 'GPT-4o-mini',
            'Technique': technique,
            'Metric': 'EX',
            'With RAG': f"{gpt_metrics['with_rag_ex_pct']:.1f}%",
            'Without RAG': f"{gpt_metrics['without_rag_ex_pct']:.1f}%",
            'Improvement': f"+{gpt_metrics['with_rag_ex_pct'] - gpt_metrics['without_rag_ex_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'GPT-4o-mini',
            'Technique': technique,
            'Metric': 'SM',
            'With RAG': f"{gpt_metrics['with_rag_sm_pct']:.1f}%",
            'Without RAG': f"{gpt_metrics['without_rag_sm_pct']:.1f}%",
            'Improvement': f"+{gpt_metrics['with_rag_sm_pct'] - gpt_metrics['without_rag_sm_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'GPT-4o-mini',
            'Technique': technique,
            'Metric': 'F1',
            'With RAG': f"{gpt_metrics['with_rag_f1']:.3f}",
            'Without RAG': f"{gpt_metrics['without_rag_f1']:.3f}",
            'Improvement': f"+{gpt_metrics['with_rag_f1'] - gpt_metrics['without_rag_f1']:.3f}"
        })
    
    df_table = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_file = output_dir / "table_2_technique_comparison.csv"
    df_table.to_csv(csv_file, index=False)
    print(f"\n✓ Table 2 saved: {csv_file}")
    
    # Also save as formatted text
    txt_file = output_dir / "table_2_technique_comparison.txt"
    with open(txt_file, 'w') as f:
        f.write("Table 2: Technique-by-Technique Comparison\n")
        f.write("="*80 + "\n\n")
        f.write(df_table.to_string(index=False))
    print(f"✓ Table 2 (text) saved: {txt_file}")
    
    return df_table


def create_figure_1_ollama_only(metrics_ollama: Dict, output_dir: Path):
    """Create Figure 1 for Ollama only (single subplot)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['Success Rate', 'EX', 'SM', 'F1-Score']
    x = range(len(metrics))
    width = 0.35
    
    ollama_with = [
        metrics_ollama['with_rag_success_pct'],
        metrics_ollama['with_rag_ex_pct'],
        metrics_ollama['with_rag_sm_pct'],
        metrics_ollama['with_rag_f1'] * 100
    ]
    ollama_without = [
        metrics_ollama['without_rag_success_pct'],
        metrics_ollama['without_rag_ex_pct'],
        metrics_ollama['without_rag_sm_pct'],
        metrics_ollama['without_rag_f1'] * 100
    ]
    
    ax.bar([i - width/2 for i in x], ollama_with, width, label='With RAG', color=COLORS['with_rag'])
    ax.bar([i + width/2 for i in x], ollama_without, width, label='Without RAG', color=COLORS['without_rag'])
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Percentage / Score', fontweight='bold')
    ax.set_title('Ollama (llama3.1:8b): Overall Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = output_dir / "figure_1_overall_metrics_ollama_only.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 1 (Ollama only) saved: {output_file}")
    plt.close()


def create_table_2_ollama_only(results_ollama: Dict[str, pd.DataFrame], output_dir: Path):
    """Create Table 2 for Ollama only."""
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    table_data = []
    
    for technique in techniques:
        ollama_df = results_ollama[technique]
        ollama_metrics = calculate_overall_metrics(ollama_df)
        
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'EX',
            'With RAG': f"{ollama_metrics['with_rag_ex_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_ex_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_ex_pct'] - ollama_metrics['without_rag_ex_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'SM',
            'With RAG': f"{ollama_metrics['with_rag_sm_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_sm_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_sm_pct'] - ollama_metrics['without_rag_sm_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Ollama',
            'Technique': technique,
            'Metric': 'F1',
            'With RAG': f"{ollama_metrics['with_rag_f1']:.3f}",
            'Without RAG': f"{ollama_metrics['without_rag_f1']:.3f}",
            'Improvement': f"+{ollama_metrics['with_rag_f1'] - ollama_metrics['without_rag_f1']:.3f}"
        })
    
    df_table = pd.DataFrame(table_data)
    csv_file = output_dir / "table_2_technique_comparison_ollama_only.csv"
    df_table.to_csv(csv_file, index=False)
    print(f"\n✓ Table 2 (Ollama only) saved: {csv_file}")
    return df_table


def create_figure_6_ollama_only(results_ollama: Dict[str, pd.DataFrame], output_dir: Path):
    """Create Figure 6 for Ollama only."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    x = range(len(techniques))
    width = 0.6
    
    ollama_with_rag = []
    ollama_without_rag = []
    ollama_failed = []
    
    for technique in techniques:
        df = results_ollama[technique]
        total = len(df)
        with_rag_success = sum(df['With_RAG_Success'] == '✅')
        without_rag_success = sum(df['Without_RAG_Success'] == '✅')
        ollama_with_rag.append(with_rag_success)
        ollama_without_rag.append(without_rag_success - with_rag_success)
        ollama_failed.append(total - without_rag_success)
    
    ax.bar(x, ollama_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax.bar(x, ollama_without_rag, width, bottom=ollama_with_rag, label='Success without RAG only', color='#FFB347')
    ax.bar(x, ollama_failed, width, bottom=[a+b for a, b in zip(ollama_with_rag, ollama_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax.set_xlabel('Techniques', fontweight='bold')
    ax.set_ylabel('Count of Queries', fontweight='bold')
    ax.set_title('Ollama (llama3.1:8b): Success Rate Breakdown', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "figure_6_success_breakdown_ollama_only.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 6 (Ollama only) saved: {output_file}")
    plt.close()


def create_figure_6(results_ollama: Dict[str, pd.DataFrame], results_gpt: Dict[str, pd.DataFrame], output_dir: Path):
    """Create Figure 6: Stacked bar chart - Success breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    x = range(len(techniques))
    width = 0.6
    
    # Calculate data for Ollama
    ollama_with_rag = []
    ollama_without_rag = []
    ollama_failed = []
    
    for technique in techniques:
        df = results_ollama[technique]
        total = len(df)
        with_rag_success = sum(df['With_RAG_Success'] == '✅')
        without_rag_success = sum(df['Without_RAG_Success'] == '✅')
        ollama_with_rag.append(with_rag_success)
        ollama_without_rag.append(without_rag_success - with_rag_success)  # Only those that succeeded without but not with
        ollama_failed.append(total - without_rag_success)  # Failed in both
    
    # Ollama subplot
    ax1.bar(x, ollama_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax1.bar(x, ollama_without_rag, width, bottom=ollama_with_rag, label='Success without RAG only', color='#FFB347')
    ax1.bar(x, ollama_failed, width, bottom=[a+b for a, b in zip(ollama_with_rag, ollama_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax1.set_xlabel('Techniques', fontweight='bold')
    ax1.set_ylabel('Count of Queries', fontweight='bold')
    ax1.set_title('Ollama (llama3.1:8b): Success Rate Breakdown', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(techniques)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Calculate data for GPT
    gpt_with_rag = []
    gpt_without_rag = []
    gpt_failed = []
    
    for technique in techniques:
        df = results_gpt[technique]
        total = len(df)
        with_rag_success = sum(df['With_RAG_Success'] == '✅')
        without_rag_success = sum(df['Without_RAG_Success'] == '✅')
        gpt_with_rag.append(with_rag_success)
        gpt_without_rag.append(without_rag_success - with_rag_success)
        gpt_failed.append(total - without_rag_success)
    
    # GPT subplot
    ax2.bar(x, gpt_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax2.bar(x, gpt_without_rag, width, bottom=gpt_with_rag, label='Success without RAG only', color='#FFB347')
    ax2.bar(x, gpt_failed, width, bottom=[a+b for a, b in zip(gpt_with_rag, gpt_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax2.set_xlabel('Techniques', fontweight='bold')
    ax2.set_ylabel('Count of Queries', fontweight='bold')
    ax2.set_title('GPT-4o-mini: Success Rate Breakdown', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(techniques)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "figure_6_success_breakdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 6 saved: {output_file}")
    plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION: 100 Test Cases")
    print("Models: Ollama (llama3.1:8b) & GPT-4o-mini")
    print("="*80)
    
    # Configuration
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    # Fix database path - check if it exists in data/ or root
    db_path = parent_dir / "data" / "income.db"
    if not db_path.exists():
        db_path = parent_dir / "income.db"
    db_url = os.getenv("DB_URL", f"sqlite:///{db_path}")
    
    # Create output directory
    output_dir = script_dir / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 100 test cases
    test_cases = load_all_100_test_cases()
    
    if len(test_cases) < 100:
        print(f"Warning: Only {len(test_cases)} test cases available. Generating additional cases...")
        additional = generate_50_additional_test_cases()
        test_cases.extend(additional[:100 - len(test_cases)])
    
    print(f"\nTotal test cases: {len(test_cases)}")
    
    # Save all 100 test cases
    test_cases_file = output_dir / "all_100_test_cases.json"
    with open(test_cases_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2)
    print(f"Saved all test cases to: {test_cases_file}\n")
    
    # Run evaluation for Ollama
    print("\n" + "="*80)
    print("Starting Ollama Evaluation")
    print("="*80)
    results_ollama = run_evaluation_for_model(
        "ollama/llama3.1:8b",
        test_cases,
        techniques,
        db_url,
        output_dir
    )
    
    # Run evaluation for GPT-4o-mini (check if API key exists)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("\n" + "="*80)
        print("⚠️  WARNING: OPENAI_API_KEY not found in .env")
        print("Skipping GPT-4o-mini evaluation")
        print("="*80)
        print("\nTo enable GPT evaluation, add to .env file:")
        print("OPENAI_API_KEY=sk-your-key-here")
        print("\nGenerating visualizations with Ollama data only...")
        results_gpt = None
    else:
        print("\n" + "="*80)
        print("Starting GPT-4o-mini Evaluation")
        print("="*80)
        results_gpt = run_evaluation_for_model(
            "openai/gpt-4o-mini",
            test_cases,
            techniques,
            db_url,
            output_dir
        )
    
    # Calculate overall metrics (averaged across all techniques)
    print("\n" + "="*80)
    print("CALCULATING OVERALL METRICS")
    print("="*80)
    
    # Combine all techniques for overall metrics
    all_ollama = pd.concat([results_ollama[t] for t in techniques], ignore_index=True)
    metrics_ollama = calculate_overall_metrics(all_ollama)
    
    if results_gpt:
        all_gpt = pd.concat([results_gpt[t] for t in techniques], ignore_index=True)
        metrics_gpt = calculate_overall_metrics(all_gpt)
    else:
        metrics_gpt = None
    
    print(f"\nOllama Overall Metrics:")
    print(f"  Success Rate: {metrics_ollama['with_rag_success_pct']:.1f}% (with) vs {metrics_ollama['without_rag_success_pct']:.1f}% (without)")
    print(f"  EX: {metrics_ollama['with_rag_ex_pct']:.1f}% (with) vs {metrics_ollama['without_rag_ex_pct']:.1f}% (without)")
    print(f"  SM: {metrics_ollama['with_rag_sm_pct']:.1f}% (with) vs {metrics_ollama['without_rag_sm_pct']:.1f}% (without)")
    print(f"  F1: {metrics_ollama['with_rag_f1']:.3f} (with) vs {metrics_ollama['without_rag_f1']:.3f} (without)")
    
    if metrics_gpt:
        print(f"\nGPT-4o-mini Overall Metrics:")
        print(f"  Success Rate: {metrics_gpt['with_rag_success_pct']:.1f}% (with) vs {metrics_gpt['without_rag_success_pct']:.1f}% (without)")
        print(f"  EX: {metrics_gpt['with_rag_ex_pct']:.1f}% (with) vs {metrics_gpt['without_rag_ex_pct']:.1f}% (without)")
        print(f"  SM: {metrics_gpt['with_rag_sm_pct']:.1f}% (with) vs {metrics_gpt['without_rag_sm_pct']:.1f}% (without)")
        print(f"  F1: {metrics_gpt['with_rag_f1']:.3f} (with) vs {metrics_gpt['without_rag_f1']:.3f} (without)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    if metrics_gpt:
        create_figure_1(metrics_ollama, metrics_gpt, output_dir)
        table_2 = create_table_2(results_ollama, results_gpt, output_dir)
        create_figure_6(results_ollama, results_gpt, output_dir)
    else:
        print("\n⚠️  Skipping GPT visualizations (no GPT data available)")
        print("Generating Ollama-only visualizations...")
        # Create single-model versions
        create_figure_1_ollama_only(metrics_ollama, output_dir)
        table_2 = create_table_2_ollama_only(results_ollama, output_dir)
        create_figure_6_ollama_only(results_ollama, output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - figure_1_overall_metrics.png")
    print("  - table_2_technique_comparison.csv")
    print("  - table_2_technique_comparison.txt")
    print("  - figure_6_success_breakdown.png")
    print("  - Individual technique results CSV files")


if __name__ == "__main__":
    main()

