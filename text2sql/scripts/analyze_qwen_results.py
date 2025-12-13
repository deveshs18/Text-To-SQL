"""Analyze Qwen evaluation results to identify which queries passed."""
import pandas as pd
from pathlib import Path

# Load all Qwen results
techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']
dfs = {}
for t in techniques:
    csv_path = Path(f'text2sql/scripts/data/results/ollama_qwen-0.5b-spider_{t}_results.csv')
    if csv_path.exists():
        dfs[t] = pd.read_csv(csv_path)

if not dfs:
    print("No results files found!")
    exit(1)

# Combine all results
all_df = pd.concat(dfs.values(), ignore_index=True)

# Filter passed queries
passed_with_rag = all_df[all_df['With_RAG_EX'] == '✅']
passed_without_rag = all_df[all_df['Without_RAG_EX'] == '✅']
failed_with_rag = all_df[all_df['With_RAG_EX'] == '❌']

print("="*80)
print("QWEN MODEL EVALUATION RESULTS ANALYSIS")
print("="*80)
print(f"\nTotal queries evaluated: {len(all_df)}")
print(f"Passed WITH RAG (EX): {len(passed_with_rag)}/{len(all_df)} ({len(passed_with_rag)/len(all_df)*100:.1f}%)")
print(f"Passed WITHOUT RAG (EX): {len(passed_without_rag)}/{len(all_df)} ({len(passed_without_rag)/len(all_df)*100:.1f}%)")
print(f"Failed WITH RAG (EX): {len(failed_with_rag)}/{len(all_df)} ({len(failed_with_rag)/len(all_df)*100:.1f}%)")

print("\n" + "="*80)
print("QUERIES THAT PASSED WITH RAG (Execution Accuracy)")
print("="*80)
for i, (idx, row) in enumerate(passed_with_rag.iterrows(), 1):
    print(f"{i}. {row['Question']}")

print("\n" + "="*80)
print("QUERIES THAT FAILED WITH RAG (Execution Accuracy)")
print("="*80)
for i, (idx, row) in enumerate(failed_with_rag.iterrows(), 1):
    print(f"{i}. {row['Question']}")

# Analyze patterns
print("\n" + "="*80)
print("PATTERN ANALYSIS")
print("="*80)

# Simple queries (COUNT, AVG, basic SELECT)
simple_keywords = ['count', 'how many', 'average', 'avg', 'show', 'list']
# Complex queries (GROUP BY, HAVING, CASE, CTE, window functions)
complex_keywords = ['percentage', 'each', 'for each', 'group', 'having', 'top', 'most common']

passed_simple = sum(1 for q in passed_with_rag['Question'].str.lower() 
                    if any(kw in q for kw in simple_keywords))
failed_simple = sum(1 for q in failed_with_rag['Question'].str.lower() 
                    if any(kw in q for kw in simple_keywords))

passed_complex = sum(1 for q in passed_with_rag['Question'].str.lower() 
                     if any(kw in q for kw in complex_keywords))
failed_complex = sum(1 for q in failed_with_rag['Question'].str.lower() 
                     if any(kw in q for kw in complex_keywords))

print(f"\nSimple queries (COUNT, AVG, basic SELECT):")
print(f"  Passed: {passed_simple}")
print(f"  Failed: {failed_simple}")

print(f"\nComplex queries (GROUP BY, percentage, each, etc.):")
print(f"  Passed: {passed_complex}")
print(f"  Failed: {failed_complex}")


