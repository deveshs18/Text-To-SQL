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
    """Generate 35 test cases (15 easy + 10 easy-medium + 10 medium)."""
    return [
        # =========================
        # 15 EASY QUERIES
        # =========================
        {
            "question": "How many records are there in the adult_income table?",
            "gold_sql": "SELECT COUNT(*) AS total_rows FROM adult_income;"
        },
        {
            "question": "How many people earn more than 50K?",
            "gold_sql": "SELECT COUNT(*) AS high_earners FROM adult_income WHERE income = '>50K';"
        },
        {
            "question": "What is the average age of all people?",
            "gold_sql": "SELECT AVG(age) AS avg_age FROM adult_income;"
        },
        {
            "question": "What is the minimum and maximum age in the dataset?",
            "gold_sql": "SELECT MIN(age) AS min_age, MAX(age) AS max_age FROM adult_income;"
        },
        {
            "question": "What is the average number of hours worked per week?",
            "gold_sql": "SELECT AVG(hours_per_week) AS avg_hours_per_week FROM adult_income;"
        },
        {
            "question": "How many women are in the dataset?",
            "gold_sql": "SELECT COUNT(*) AS total_women FROM adult_income WHERE sex = 'Female';"
        },
        {
            "question": "How many men are in the dataset?",
            "gold_sql": "SELECT COUNT(*) AS total_men FROM adult_income WHERE sex = 'Male';"
        },
        {
            "question": "How many people have the native country as United-States?",
            "gold_sql": "SELECT COUNT(*) AS total_us FROM adult_income WHERE native_country = 'United-States';"
        },
        {
            "question": "How many people have a capital_gain greater than zero?",
            "gold_sql": "SELECT COUNT(*) AS people_with_gain FROM adult_income WHERE capital_gain > 0;"
        },
        {
            "question": "How many people have a capital_loss greater than zero?",
            "gold_sql": "SELECT COUNT(*) AS people_with_loss FROM adult_income WHERE capital_loss > 0;"
        },
        {
            "question": "List all distinct education levels in the dataset.",
            "gold_sql": "SELECT DISTINCT education FROM adult_income ORDER BY education;"
        },
        {
            "question": "List all distinct occupations in the dataset (including '?').",
            "gold_sql": "SELECT DISTINCT occupation FROM adult_income ORDER BY occupation;"
        },
        {
            "question": "Show the top 5 oldest people with their age, sex, and workclass.",
            "gold_sql": "SELECT age, sex, workclass FROM adult_income ORDER BY age DESC LIMIT 5;"
        },
        {
            "question": "Show the 5 people who work the fewest hours per week, with their age and hours_per_week.",
            "gold_sql": "SELECT age, hours_per_week FROM adult_income ORDER BY hours_per_week ASC LIMIT 5;"
        },
        {
            "question": "How many people are between 30 and 40 years old (inclusive)?",
            "gold_sql": "SELECT COUNT(*) AS people_30_to_40 FROM adult_income WHERE age BETWEEN 30 AND 40;"
        },
        # =========================
        # 10 EASY–MEDIUM QUERIES
        # =========================
        {
            "question": "What is the average number of hours worked per week for men and women?",
            "gold_sql": "SELECT sex, AVG(hours_per_week) AS avg_hours_per_week FROM adult_income GROUP BY sex ORDER BY sex;"
        },
        {
            "question": "How many people are there in each income group (<=50K and >50K)?",
            "gold_sql": "SELECT income, COUNT(*) AS total_people FROM adult_income GROUP BY income ORDER BY income;"
        },
        {
            "question": "For each education level, what is the average education_num?",
            "gold_sql": "SELECT education, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY education ORDER BY avg_education_num DESC;"
        },
        {
            "question": "For each workclass, how many people are there?",
            "gold_sql": "SELECT workclass, COUNT(*) AS total_people FROM adult_income GROUP BY workclass ORDER BY total_people DESC;"
        },
        {
            "question": "For each race, what is the average age?",
            "gold_sql": "SELECT race, AVG(age) AS avg_age FROM adult_income GROUP BY race ORDER BY avg_age DESC;"
        },
        {
            "question": "What is the average capital_gain for each income group?",
            "gold_sql": "SELECT income, AVG(capital_gain) AS avg_capital_gain FROM adult_income GROUP BY income ORDER BY income;"
        },
        {
            "question": "For each marital status, how many people earn more than 50K?",
            "gold_sql": "SELECT marital_status, COUNT(*) AS high_earners FROM adult_income WHERE income = '>50K' GROUP BY marital_status ORDER BY high_earners DESC;"
        },
        {
            "question": "For each native country, how many people are there? Show the top 10 countries by count.",
            "gold_sql": "SELECT native_country, COUNT(*) AS total_people FROM adult_income GROUP BY native_country ORDER BY total_people DESC LIMIT 10;"
        },
        {
            "question": "What is the average age of people who earn more than 50K?",
            "gold_sql": "SELECT AVG(age) AS avg_age FROM adult_income WHERE income = '>50K';"
        },
        {
            "question": "What is the average age of people who earn less than or equal to 50K?",
            "gold_sql": "SELECT AVG(age) AS avg_age FROM adult_income WHERE income = '<=50K';"
        },
        {
            "question": "What is the average number of hours worked per week by women?",
            "gold_sql": "SELECT AVG(hours_per_week) AS avg_hours_per_week FROM adult_income WHERE sex = 'Female';"
        },
        {
            "question": "What is the average number of hours worked per week by men?",
            "gold_sql": "SELECT AVG(hours_per_week) AS avg_hours_per_week FROM adult_income WHERE sex = 'Male';"
        },
        # =========================
        # 10 MEDIUM(-ISH) QUERIES (Simplified)
        # =========================
        {
            "question": "What is the average education_num of all people in the dataset?",
            "gold_sql": "SELECT AVG(education_num) AS avg_education_num FROM adult_income;"
        },
        {
            "question": "What is the maximum capital_gain in the dataset?",
            "gold_sql": "SELECT MAX(capital_gain) AS max_capital_gain FROM adult_income;"
        },
        {
            "question": "What is the maximum capital_loss in the dataset?",
            "gold_sql": "SELECT MAX(capital_loss) AS max_capital_loss FROM adult_income;"
        },
        {
            "question": "What is the average capital_gain for people who have capital_gain greater than zero?",
            "gold_sql": "SELECT AVG(capital_gain) AS avg_capital_gain FROM adult_income WHERE capital_gain > 0;"
        },
        {
            "question": "What is the average capital_loss for people who have capital_loss greater than zero?",
            "gold_sql": "SELECT AVG(capital_loss) AS avg_capital_loss FROM adult_income WHERE capital_loss > 0;"
        },
        {
            "question": "What is the average age of people who work more than 50 hours per week?",
            "gold_sql": "SELECT AVG(age) AS avg_age FROM adult_income WHERE hours_per_week > 50;"
        },
        {
            "question": "How many people have education = 'Bachelors'?",
            "gold_sql": "SELECT COUNT(*) AS total_bachelors FROM adult_income WHERE education = 'Bachelors';"
        },
        {
            "question": "How many people have marital_status = 'Never-married'?",
            "gold_sql": "SELECT COUNT(*) AS total_never_married FROM adult_income WHERE marital_status = 'Never-married';"
        },
        {
            "question": "How many people have occupation = 'Exec-managerial'?",
            "gold_sql": "SELECT COUNT(*) AS total_exec_managerial FROM adult_income WHERE occupation = 'Exec-managerial';"
        },
        {
            "question": "How many people have native_country not equal to 'United-States'?",
            "gold_sql": "SELECT COUNT(*) AS total_non_us FROM adult_income WHERE native_country <> 'United-States';"
        },
        {
            "question": "How many people have both capital_gain and capital_loss equal to zero?",
            "gold_sql": "SELECT COUNT(*) AS total_no_gain_loss FROM adult_income WHERE capital_gain = 0 AND capital_loss = 0;"
        },
        {
            "question": "How many people have hours_per_week greater than or equal to 60?",
            "gold_sql": "SELECT COUNT(*) AS total_60plus_hours FROM adult_income WHERE hours_per_week >= 60;"
        },
        {
            "question": "How many people have education_num greater than or equal to 13?",
            "gold_sql": "SELECT COUNT(*) AS total_high_education FROM adult_income WHERE education_num >= 13;"
        },
        {
            "question": "Show 5 women who work more than 50 hours per week, with their age and occupation.",
            "gold_sql": "SELECT age, occupation FROM adult_income WHERE sex = 'Female' AND hours_per_week > 50 LIMIT 5;"
        },
        {
            "question": "Show 5 men older than 60 years, with their age, marital_status, and income.",
            "gold_sql": "SELECT age, marital_status, income FROM adult_income WHERE sex = 'Male' AND age > 60 LIMIT 5;"
        },
        {
            "question": "Show 10 people who have capital_gain greater than zero, with their age, sex, and capital_gain.",
            "gold_sql": "SELECT age, sex, capital_gain FROM adult_income WHERE capital_gain > 0 LIMIT 10;"
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
    
    # Latency (extract numeric values, skip "SKIPPED" entries)
    with_rag_latency_series = df[df['With_RAG_Latency'] != 'SKIPPED']['With_RAG_Latency'].str.replace('s', '').astype(float)
    without_rag_latency_series = df[df['Without_RAG_Latency'] != 'SKIPPED']['Without_RAG_Latency'].str.replace('s', '').astype(float)
    with_rag_latency = with_rag_latency_series.mean() if len(with_rag_latency_series) > 0 else 0.0
    without_rag_latency = without_rag_latency_series.mean() if len(without_rag_latency_series) > 0 else 0.0
    
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
        'without_rag_ex_count': without_rag_ex,
        'with_rag_sm_count': with_rag_sm,
        'without_rag_sm_count': without_rag_sm
    }


def create_figure_1(metrics_arctic: Dict, metrics_qwen: Dict = None, metrics_gpt: Dict = None, output_dir: Path = None):
    """Create Figure 1: Overall metrics comparison bar chart for 2-3 models."""
    # Determine how many models we have
    has_qwen = metrics_qwen is not None
    has_gpt = metrics_gpt is not None
    
    if has_qwen and has_gpt:
        # Three models: Arctic, Qwen, GPT
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    elif has_qwen or has_gpt:
        # Two models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        # Single model
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None
        ax3 = None
    
    metrics = ['Success Rate', 'EX', 'SM', 'F1-Score']
    x = range(len(metrics))
    width = 0.35
    
    # Arctic subplot
    arctic_with = [
        metrics_arctic['with_rag_success_pct'],
        metrics_arctic['with_rag_ex_pct'],
        metrics_arctic['with_rag_sm_pct'],
        metrics_arctic['with_rag_f1'] * 100
    ]
    arctic_without = [
        metrics_arctic['without_rag_success_pct'],
        metrics_arctic['without_rag_ex_pct'],
        metrics_arctic['without_rag_sm_pct'],
        metrics_arctic['without_rag_f1'] * 100
    ]
    
    ax1.bar([i - width/2 for i in x], arctic_with, width, label='With RAG', color=COLORS['with_rag'])
    ax1.bar([i + width/2 for i in x], arctic_without, width, label='Without RAG', color=COLORS['without_rag'])
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('Arctic Base Model: Overall Performance', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Qwen subplot (if available)
    if has_qwen and ax2:
        qwen_with = [
            metrics_qwen['with_rag_success_pct'],
            metrics_qwen['with_rag_ex_pct'],
            metrics_qwen['with_rag_sm_pct'],
            metrics_qwen['with_rag_f1'] * 100
        ]
        qwen_without = [
            metrics_qwen['without_rag_success_pct'],
            metrics_qwen['without_rag_ex_pct'],
            metrics_qwen['without_rag_sm_pct'],
            metrics_qwen['without_rag_f1'] * 100
        ]
        
        ax2.bar([i - width/2 for i in x], qwen_with, width, label='With RAG', color=COLORS['with_rag'])
        ax2.bar([i + width/2 for i in x], qwen_without, width, label='Without RAG', color=COLORS['without_rag'])
        ax2.set_xlabel('Metrics', fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontweight='bold')
        ax2.set_title('Qwen-0.5B-Spider: Overall Performance', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
    
    # GPT subplot (if available)
    if has_gpt:
        gpt_ax = ax3 if has_qwen else ax2
        if gpt_ax:
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
            
            gpt_ax.bar([i - width/2 for i in x], gpt_with, width, label='With RAG', color=COLORS['with_rag'])
            gpt_ax.bar([i + width/2 for i in x], gpt_without, width, label='Without RAG', color=COLORS['without_rag'])
            gpt_ax.set_xlabel('Metrics', fontweight='bold')
            gpt_ax.set_ylabel('Percentage (%)', fontweight='bold')
            gpt_ax.set_title('GPT-4o-mini: Overall Performance', fontweight='bold', fontsize=12)
            gpt_ax.set_xticks(x)
            gpt_ax.set_xticklabels(metrics)
            gpt_ax.legend()
            gpt_ax.grid(axis='y', alpha=0.3)
            gpt_ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = output_dir / "figure_1_overall_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 1 saved: {output_file}")
    plt.close()


def create_table_2(results_arctic: Dict[str, pd.DataFrame], results_qwen: Dict[str, pd.DataFrame] = None, results_gpt: Dict[str, pd.DataFrame] = None, output_dir: Path = None):
    """Create Table 2: Technique-by-technique comparison for 2-3 models."""
    techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
    table_data = []
    
    for technique in techniques:
        # Arctic metrics
        arctic_df = results_arctic[technique]
        arctic_metrics = calculate_overall_metrics(arctic_df)
        
        # Add rows for Arctic
        table_data.append({
            'Model': 'Arctic Base',
            'Technique': technique,
            'Metric': 'EX',
            'With RAG': f"{arctic_metrics['with_rag_ex_pct']:.1f}%",
            'Without RAG': f"{arctic_metrics['without_rag_ex_pct']:.1f}%",
            'Improvement': f"+{arctic_metrics['with_rag_ex_pct'] - arctic_metrics['without_rag_ex_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Arctic Base',
            'Technique': technique,
            'Metric': 'SM',
            'With RAG': f"{arctic_metrics['with_rag_sm_pct']:.1f}%",
            'Without RAG': f"{arctic_metrics['without_rag_sm_pct']:.1f}%",
            'Improvement': f"+{arctic_metrics['with_rag_sm_pct'] - arctic_metrics['without_rag_sm_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Arctic Base',
            'Technique': technique,
            'Metric': 'F1',
            'With RAG': f"{arctic_metrics['with_rag_f1'] * 100:.1f}%",
            'Without RAG': f"{arctic_metrics['without_rag_f1'] * 100:.1f}%",
            'Improvement': f"+{(arctic_metrics['with_rag_f1'] - arctic_metrics['without_rag_f1']) * 100:.1f}%"
        })
        
        # Add rows for Qwen (if available)
        if results_qwen:
            qwen_df = results_qwen[technique]
            qwen_metrics = calculate_overall_metrics(qwen_df)
            
            table_data.append({
                'Model': 'Qwen-0.5B-Spider',
                'Technique': technique,
                'Metric': 'EX',
                'With RAG': f"{qwen_metrics['with_rag_ex_pct']:.1f}%",
                'Without RAG': f"{qwen_metrics['without_rag_ex_pct']:.1f}%",
                'Improvement': f"+{qwen_metrics['with_rag_ex_pct'] - qwen_metrics['without_rag_ex_pct']:.1f}%"
            })
            table_data.append({
                'Model': 'Qwen-0.5B-Spider',
                'Technique': technique,
                'Metric': 'SM',
                'With RAG': f"{qwen_metrics['with_rag_sm_pct']:.1f}%",
                'Without RAG': f"{qwen_metrics['without_rag_sm_pct']:.1f}%",
                'Improvement': f"+{qwen_metrics['with_rag_sm_pct'] - qwen_metrics['without_rag_sm_pct']:.1f}%"
            })
            table_data.append({
                'Model': 'Qwen-0.5B-Spider',
                'Technique': technique,
                'Metric': 'F1',
                'With RAG': f"{qwen_metrics['with_rag_f1'] * 100:.1f}%",
                'Without RAG': f"{qwen_metrics['without_rag_f1'] * 100:.1f}%",
                'Improvement': f"+{(qwen_metrics['with_rag_f1'] - qwen_metrics['without_rag_f1']) * 100:.1f}%"
            })
        
        # Add rows for GPT (if available)
        if results_gpt:
            gpt_df = results_gpt[technique]
            gpt_metrics = calculate_overall_metrics(gpt_df)
            
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
                'With RAG': f"{gpt_metrics['with_rag_f1'] * 100:.1f}%",
                'Without RAG': f"{gpt_metrics['without_rag_f1'] * 100:.1f}%",
                'Improvement': f"+{(gpt_metrics['with_rag_f1'] - gpt_metrics['without_rag_f1']) * 100:.1f}%"
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
    ax.set_title('Arctic Base Model: Overall Performance', fontweight='bold', fontsize=14)
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
            'Model': 'Arctic Fine-tuned',
            'Technique': technique,
            'Metric': 'EX',
            'With RAG': f"{ollama_metrics['with_rag_ex_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_ex_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_ex_pct'] - ollama_metrics['without_rag_ex_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Arctic Fine-tuned',
            'Technique': technique,
            'Metric': 'SM',
            'With RAG': f"{ollama_metrics['with_rag_sm_pct']:.1f}%",
            'Without RAG': f"{ollama_metrics['without_rag_sm_pct']:.1f}%",
            'Improvement': f"+{ollama_metrics['with_rag_sm_pct'] - ollama_metrics['without_rag_sm_pct']:.1f}%"
        })
        table_data.append({
            'Model': 'Arctic Fine-tuned',
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
        # Convert to percentages
        ollama_with_rag.append((with_rag_success / total) * 100)
        ollama_without_rag.append(((without_rag_success - with_rag_success) / total) * 100)
        ollama_failed.append(((total - without_rag_success) / total) * 100)
    
    ax.bar(x, ollama_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax.bar(x, ollama_without_rag, width, bottom=ollama_with_rag, label='Success without RAG only', color='#FFB347')
    ax.bar(x, ollama_failed, width, bottom=[a+b for a, b in zip(ollama_with_rag, ollama_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax.set_xlabel('Techniques', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Arctic Base Model: Success Rate Breakdown', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 100)
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
        # Convert to percentages
        ollama_with_rag.append((with_rag_success / total) * 100)
        ollama_without_rag.append(((without_rag_success - with_rag_success) / total) * 100)  # Only those that succeeded without but not with
        ollama_failed.append(((total - without_rag_success) / total) * 100)  # Failed in both
    
    # Ollama subplot
    ax1.bar(x, ollama_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax1.bar(x, ollama_without_rag, width, bottom=ollama_with_rag, label='Success without RAG only', color='#FFB347')
    ax1.bar(x, ollama_failed, width, bottom=[a+b for a, b in zip(ollama_with_rag, ollama_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax1.set_xlabel('Techniques', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('Arctic Base Model: Success Rate Breakdown', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
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
        # Convert to percentages
        gpt_with_rag.append((with_rag_success / total) * 100)
        gpt_without_rag.append(((without_rag_success - with_rag_success) / total) * 100)
        gpt_failed.append(((total - without_rag_success) / total) * 100)
    
    # GPT subplot
    ax2.bar(x, gpt_with_rag, width, label='Success with RAG', color=COLORS['with_rag'])
    ax2.bar(x, gpt_without_rag, width, bottom=gpt_with_rag, label='Success without RAG only', color='#FFB347')
    ax2.bar(x, gpt_failed, width, bottom=[a+b for a, b in zip(gpt_with_rag, gpt_without_rag)], 
            label='Failed', color='#D3D3D3')
    ax2.set_xlabel('Techniques', fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_title('GPT-4o-mini: Success Rate Breakdown', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 100)
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
    print("COMPREHENSIVE EVALUATION")
    print("Models: Arctic Base Model, Qwen-0.5B-Spider & GPT-4o-mini")
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
    
    # Load test cases
    NUM_TEST_CASES = 35
    test_cases = load_all_100_test_cases()
    
    # Take only first NUM_TEST_CASES
    if len(test_cases) >= NUM_TEST_CASES:
        test_cases = test_cases[:NUM_TEST_CASES]
    else:
        print(f"Warning: Only {len(test_cases)} test cases available. Using all available.")
    
    # Save test cases
    test_cases_file = output_dir / f"all_{NUM_TEST_CASES}_test_cases.json"
    with open(test_cases_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2)
    print(f"Saved all test cases to: {test_cases_file}\n")
    
    # Run evaluation for Arctic Base Model
    print("\n" + "="*80)
    print("Starting Arctic Base Model Evaluation")
    print("="*80)
    print("WARNING: Make sure arctic_base_server.py is running on port 11437!")
    print("="*80)
    results_arctic = run_evaluation_for_model(
        "ollama/arctic-base",
        test_cases,
        techniques,
        db_url,
        output_dir
    )
    
    # Run evaluation for Qwen-0.5B-Spider
    print("\n" + "="*80)
    print("Starting Qwen-0.5B-Spider Evaluation")
    print("="*80)
    print("WARNING: Make sure qwen_server.py is running on port 11438!")
    print("="*80)
    results_qwen = run_evaluation_for_model(
        "ollama/qwen-0.5b-spider",
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
    all_arctic = pd.concat([results_arctic[t] for t in techniques], ignore_index=True)
    metrics_arctic = calculate_overall_metrics(all_arctic)
    
    all_qwen = pd.concat([results_qwen[t] for t in techniques], ignore_index=True)
    metrics_qwen = calculate_overall_metrics(all_qwen)
    
    if results_gpt:
        all_gpt = pd.concat([results_gpt[t] for t in techniques], ignore_index=True)
        metrics_gpt = calculate_overall_metrics(all_gpt)
    else:
        metrics_gpt = None
    
    print(f"\nArctic Base Model Overall Metrics:")
    print(f"  Success Rate: {metrics_arctic['with_rag_success_pct']:.1f}% (with) vs {metrics_arctic['without_rag_success_pct']:.1f}% (without)")
    print(f"  EX: {metrics_arctic['with_rag_ex_pct']:.1f}% (with) vs {metrics_arctic['without_rag_ex_pct']:.1f}% (without)")
    print(f"  SM: {metrics_arctic['with_rag_sm_pct']:.1f}% (with) vs {metrics_arctic['without_rag_sm_pct']:.1f}% (without)")
    
    print(f"\nQwen-0.5B-Spider Overall Metrics:")
    print(f"  Success Rate: {metrics_qwen['with_rag_success_pct']:.1f}% (with) vs {metrics_qwen['without_rag_success_pct']:.1f}% (without)")
    print(f"  EX: {metrics_qwen['with_rag_ex_pct']:.1f}% (with) vs {metrics_qwen['without_rag_ex_pct']:.1f}% (without)")
    print(f"  SM: {metrics_qwen['with_rag_sm_pct']:.1f}% (with) vs {metrics_qwen['without_rag_sm_pct']:.1f}% (without)")
    print(f"  F1: {metrics_qwen['with_rag_f1']:.3f} (with) vs {metrics_qwen['without_rag_f1']:.3f} (without)")
    print(f"\n✅ Qwen Execution Accuracy (With RAG): {metrics_qwen['with_rag_ex_pct']:.1f}%")
    print(f"✅ Qwen Execution Accuracy (Without RAG): {metrics_qwen['without_rag_ex_pct']:.1f}%")
    
    print(f"\n✅ Arctic Execution Accuracy (With RAG): {metrics_arctic['with_rag_ex_pct']:.1f}%")
    print(f"✅ Arctic Execution Accuracy (Without RAG): {metrics_arctic['without_rag_ex_pct']:.1f}%")
    
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
    
    # Generate visualizations with all available models (all values in percent)
    create_figure_1(metrics_arctic, metrics_qwen, metrics_gpt, output_dir)
    table_2 = create_table_2(results_arctic, results_qwen, results_gpt, output_dir)
    
    # Create figure 6 (success breakdown) - update to handle 3 models if needed
    if metrics_gpt:
        # For now, keep the 2-model version for figure 6
        create_figure_6(results_arctic, results_gpt, output_dir)
    elif metrics_qwen:
        # Create a 2-model version with Arctic and Qwen
        create_figure_6(results_arctic, results_qwen, output_dir)
    else:
        create_figure_6_ollama_only(results_arctic, output_dir)
    
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

