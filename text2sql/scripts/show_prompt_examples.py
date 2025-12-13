"""
Show examples of all 4 prompting techniques with RAG for the Text-to-SQL project.
"""
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from db import get_engine
from schema_retriever import get_schema_snippet
from prompts import build_few_shot_prompt, build_cot_prompt, build_ltm_prompt, build_eg_prompt, get_examples_for_table

# Setup
engine = get_engine('sqlite:///text2sql/income.db')
question = "How many people earn more than 50K?"
model_name = "ollama/arctic-base"

# Get schema and examples (RAG enabled)
schema = get_schema_snippet(question, engine, 'adult_income', model_name)
examples = get_examples_for_table('adult_income', simple=True)

print("="*80)
print("PROMPTING TECHNIQUES WITH RAG - EXAMPLES")
print("="*80)
print(f"\nQuestion: {question}")
print(f"Model: {model_name}")
print(f"\nRAG Schema Retrieved:")
print("-"*80)
print(schema)
print("\nRAG Examples Retrieved:")
print("-"*80)
for i, ex in enumerate(examples[:3], 1):
    print(f"Example {i}:")
    print(f"  Q: {ex['question']}")
    print(f"  SQL: {ex['sql']}")
print("="*80)

# 1. Few-Shot Prompt
print("\n\n1. FEW-SHOT (FS) PROMPT")
print("="*80)
print("Description: Provides 3-4 question→SQL examples + new question")
print("-"*80)
fs_prompt = build_few_shot_prompt(schema, question, examples, model_name)
print(fs_prompt)
print("\nKey Features:")
print("  • Includes 3-4 example Q→SQL pairs")
print("  • Shows pattern for model to follow")
print("  • New question at the end for model to answer")

# 2. Chain-of-Thought Prompt
print("\n\n2. CHAIN-OF-THOUGHT (CoT) PROMPT")
print("="*80)
print("Description: Instructs model to 'think step-by-step → then SQL only'")
print("-"*80)
cot_prompt = build_cot_prompt(schema, question, model_name)
print(cot_prompt)
print("\nKey Features:")
print("  • Explicit instruction to think step-by-step")
print("  • Guides: identify tables → columns → conditions → construct SQL")
print("  • No examples, just thinking guidance")

# 3. Least-to-Most Prompt
print("\n\n3. LEAST-TO-MOST (LtM) PROMPT")
print("="*80)
print("Description: Breaks question into substeps (tables, filters, grouping, ordering)")
print("-"*80)
ltm_prompt = build_ltm_prompt(schema, question, model_name)
print(ltm_prompt)
print("\nKey Features:")
print("  • Instructs to break down into simpler parts")
print("  • Guides: What data? → What conditions? → What grouping? → Combine")
print("  • Decomposition approach")

# 4. Execution-Guided Prompt
print("\n\n4. EXECUTION-GUIDED (EG) PROMPT")
print("="*80)
print("Description: Generate → execute → fix on error (≤3 attempts)")
print("-"*80)
eg_prompt = build_eg_prompt(schema, question, model_name)
print(eg_prompt)
print("\nKey Features:")
print("  • Instructs to verify SQL would execute correctly")
print("  • Guides: Check column names → syntax → logic")
print("  • Self-validation approach")

# Show Without RAG comparison
print("\n\n" + "="*80)
print("WITHOUT RAG COMPARISON")
print("="*80)
print("\nWithout RAG: Minimal schema (no column names, no examples)")
print("-"*80)
schema_no_rag = "adult_income(...)"
fs_no_rag = build_few_shot_prompt(schema_no_rag, question, None, model_name)
print("Few-Shot (No RAG):")
print(fs_no_rag[:200] + "...")
print("\nKey Difference:")
print("  • With RAG: Full schema with all columns + examples")
print("  • Without RAG: Minimal schema (just table name) + no examples")

print("\n" + "="*80)
print("END OF EXAMPLES")
print("="*80)


