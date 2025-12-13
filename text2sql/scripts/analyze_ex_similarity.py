"""Analyze why GPT and Arctic have same EX values across techniques."""
import pandas as pd
from pathlib import Path
from collections import defaultdict

results_dir = Path('text2sql/scripts/data/results')
techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']

def analyze_model(model_prefix, model_name):
    """Analyze which queries passed EX for each technique."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {model_name}")
    print('='*80)
    
    technique_results = {}
    
    for technique in techniques:
        file_path = results_dir / f"{model_prefix}_{technique}_results.csv"
        if not file_path.exists():
            print(f"  ❌ File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        df_filtered = df[df['With_RAG_Success'] != 'SKIPPED']
        
        # Get questions that passed EX
        passed_questions = df_filtered[df_filtered['With_RAG_EX'] == '✅']['Question'].tolist()
        failed_questions = df_filtered[df_filtered['With_RAG_EX'] == '❌']['Question'].tolist()
        
        technique_results[technique] = {
            'passed': set(passed_questions),
            'failed': set(failed_questions),
            'total': len(df_filtered),
            'passed_count': len(passed_questions)
        }
        
        print(f"\n{technique}:")
        print(f"  Passed: {len(passed_questions)}/{len(df_filtered)} ({len(passed_questions)/len(df_filtered)*100:.1f}%)")
    
    # Compare techniques
    print(f"\n{'='*80}")
    print("COMPARISON BETWEEN TECHNIQUES:")
    print('='*80)
    
    # Find queries that passed in ALL techniques
    all_passed = technique_results['Few-Shot']['passed']
    for tech in ['CoT', 'LtM', 'EG']:
        all_passed = all_passed.intersection(technique_results[tech]['passed'])
    
    print(f"\nQueries that passed EX in ALL 4 techniques: {len(all_passed)}")
    if len(all_passed) > 0:
        for q in sorted(list(all_passed))[:5]:  # Show first 5
            print(f"  - {q[:60]}...")
        if len(all_passed) > 5:
            print(f"  ... and {len(all_passed) - 5} more")
    
    # Find queries that passed in Few-Shot but failed in others
    for tech in ['CoT', 'LtM', 'EG']:
        fs_only = technique_results['Few-Shot']['passed'] - technique_results[tech]['passed']
        if len(fs_only) > 0:
            print(f"\nQueries that passed in Few-Shot but FAILED in {tech}: {len(fs_only)}")
            for q in sorted(list(fs_only))[:3]:
                print(f"  - {q[:60]}...")
        
        tech_only = technique_results[tech]['passed'] - technique_results['Few-Shot']['passed']
        if len(tech_only) > 0:
            print(f"\nQueries that passed in {tech} but FAILED in Few-Shot: {len(tech_only)}")
            for q in sorted(list(tech_only))[:3]:
                print(f"  - {q[:60]}...")
    
    # Check if all techniques have same count
    counts = [technique_results[tech]['passed_count'] for tech in techniques]
    if len(set(counts)) == 1:
        print(f"\n⚠️  ALL TECHNIQUES HAVE THE SAME EX COUNT: {counts[0]}/{technique_results['Few-Shot']['total']}")
        print("   This means the model produces identical results regardless of prompt technique!")
    else:
        print(f"\n✅ Techniques have different EX counts: {dict(zip(techniques, counts))}")
    
    return technique_results

# Analyze both models
print("="*80)
print("ANALYZING WHY GPT AND ARCTIC HAVE SAME EX VALUES")
print("="*80)

gpt_results = analyze_model('openai_gpt-4o-mini', 'GPT-4o-mini')
arctic_results = analyze_model('ollama_arctic-base', 'Arctic Base (Qwen 7B)')

# Cross-model comparison
print(f"\n{'='*80}")
print("CROSS-MODEL COMPARISON")
print('='*80)

for technique in techniques:
    gpt_passed = gpt_results[technique]['passed']
    arctic_passed = arctic_results[technique]['passed']
    
    same_passed = gpt_passed.intersection(arctic_passed)
    gpt_only = gpt_passed - arctic_passed
    arctic_only = arctic_passed - gpt_passed
    
    print(f"\n{technique}:")
    print(f"  Both models passed: {len(same_passed)}")
    print(f"  GPT only: {len(gpt_only)}")
    print(f"  Arctic only: {len(arctic_only)}")
    
    if len(gpt_only) > 0:
        print(f"  GPT-only queries:")
        for q in sorted(list(gpt_only))[:2]:
            print(f"    - {q[:50]}...")
    
    if len(arctic_only) > 0:
        print(f"  Arctic-only queries:")
        for q in sorted(list(arctic_only))[:2]:
            print(f"    - {q[:50]}...")


