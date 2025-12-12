"""
Test script to compare Qwen 0.5 base model vs Qwen 0.5 finetuned model.
Runs evaluation on both models and generates comparison results.
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Import compare_rag functions
os.chdir(parent_dir)
import importlib.util
compare_rag_path = script_dir / "compare_rag.py"
spec = importlib.util.spec_from_file_location("compare_rag", compare_rag_path)
compare_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_rag)
compare_rag_performance = compare_rag.compare_rag_performance

# Import from generate_100_evaluation
generate_eval_path = script_dir / "generate_100_evaluation.py"
spec2 = importlib.util.spec_from_file_location("generate_100_evaluation", generate_eval_path)
gen_eval = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(gen_eval)
load_all_100_test_cases = gen_eval.load_all_100_test_cases
run_evaluation_for_model = gen_eval.run_evaluation_for_model
calculate_overall_metrics = gen_eval.calculate_overall_metrics

load_dotenv()

# Configuration
NUM_TEST_CASES = 35
TECHNIQUES = ["Few-Shot", "CoT", "LtM", "EG"]


def run_model_evaluation(
    model_name: str,
    model_display_name: str,
    test_cases: List[Dict[str, str]],
    db_url: str,
    output_dir: Path,
    server_check_message: str = None
):
    """Run evaluation for a single model."""
    print("\n" + "="*80)
    print(f"STARTING {model_display_name} EVALUATION")
    print("="*80)
    
    if server_check_message:
        print(f"⚠️  {server_check_message}")
        print("="*80)
    
    print(f"Model: {model_name}")
    print(f"Test Cases: {len(test_cases)}")
    print(f"Techniques: {', '.join(TECHNIQUES)}")
    print(f"RAG Settings: With RAG & Without RAG")
    print("="*80 + "\n")
    
    # Run evaluation
    results = run_evaluation_for_model(
        model_name,
        test_cases,
        TECHNIQUES,
        db_url,
        output_dir
    )
    
    # Calculate overall metrics
    all_results = pd.concat([results[t] for t in TECHNIQUES], ignore_index=True)
    metrics = calculate_overall_metrics(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print(f"{model_display_name} - SUMMARY")
    print("="*80)
    
    print(f"\n📊 Execution Accuracy (EX):")
    print(f"   With RAG:    {metrics['with_rag_ex_count']}/{metrics['total']} ({metrics['with_rag_ex_pct']:.1f}%)")
    print(f"   Without RAG: {metrics['without_rag_ex_count']}/{metrics['total']} ({metrics['without_rag_ex_pct']:.1f}%)")
    print(f"   Improvement: {metrics['with_rag_ex_count'] - metrics['without_rag_ex_count']:+d} ({((metrics['with_rag_ex_count'] - metrics['without_rag_ex_count'])/metrics['total']*100):+.1f}%)")
    
    # Calculate average latency
    with_rag_latencies = []
    without_rag_latencies = []
    
    for technique in TECHNIQUES:
        df = results[technique]
        for lat_str in df['With_RAG_Latency']:
            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                try:
                    with_rag_latencies.append(float(str(lat_str).replace('s', '')))
                except:
                    pass
        for lat_str in df['Without_RAG_Latency']:
            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                try:
                    without_rag_latencies.append(float(str(lat_str).replace('s', '')))
                except:
                    pass
    
    avg_with_rag_latency = sum(with_rag_latencies) / len(with_rag_latencies) if with_rag_latencies else 0
    avg_without_rag_latency = sum(without_rag_latencies) / len(without_rag_latencies) if without_rag_latencies else 0
    
    print(f"\n⏱️  Average Latency:")
    print(f"   With RAG:    {avg_with_rag_latency:.2f}s")
    print(f"   Without RAG: {avg_without_rag_latency:.2f}s")
    print(f"   Difference:  {avg_with_rag_latency - avg_without_rag_latency:+.2f}s")
    
    print(f"\n✅ Success Rate:")
    print(f"   With RAG:    {metrics['with_rag_success_count']}/{metrics['total']} ({metrics['with_rag_success_pct']:.1f}%)")
    print(f"   Without RAG: {metrics['without_rag_success_count']}/{metrics['total']} ({metrics['without_rag_success_pct']:.1f}%)")
    
    print(f"\n📈 Semantic Match (SM):")
    print(f"   With RAG:    {metrics['with_rag_sm_count']}/{metrics['total']} ({metrics['with_rag_sm_pct']:.1f}%)")
    print(f"   Without RAG: {metrics['without_rag_sm_count']}/{metrics['total']} ({metrics['without_rag_sm_pct']:.1f}%)")
    
    print("="*80 + "\n")
    
    return results, metrics


def main():
    """Main function to compare Qwen base vs finetuned."""
    print("="*80)
    print("QWEN 0.5 BASELINE COMPARISON")
    print("="*80)
    print("Comparing: Qwen 0.5 Base Model vs Qwen 0.5 Finetuned Model")
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"Techniques: {', '.join(TECHNIQUES)}")
    print(f"RAG Settings: With RAG & Without RAG")
    print("="*80)
    print("\n📝 IMPORTANT NOTES:")
    print("   1. Make sure qwen_base_server.py is running on port 11439")
    print("   2. Make sure qwen_server.py is running on port 11438")
    print("   3. Each query has timeout protection (240s per case)")
    print("="*80)
    
    # Load test cases
    print("\nLoading test cases...")
    all_test_cases = load_all_100_test_cases()
    
    # Take only first NUM_TEST_CASES
    if len(all_test_cases) >= NUM_TEST_CASES:
        test_cases = all_test_cases[:NUM_TEST_CASES]
    else:
        print(f"Warning: Only {len(all_test_cases)} test cases available. Using all available.")
        test_cases = all_test_cases
    
    print(f"✅ Loaded {len(test_cases)} test cases\n")
    
    # Save test cases
    output_dir = script_dir / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases_file = output_dir / f"qwen_baseline_comparison_{NUM_TEST_CASES}_test_cases.json"
    with open(test_cases_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2)
    print(f"✅ Saved test cases to: {test_cases_file}\n")
    
    # Database setup
    db_path = script_dir / "data" / "income.db"
    if not db_path.exists():
        db_path = parent_dir / "data" / "income.db"
    if not db_path.exists():
        db_path = parent_dir / "income.db"
    db_url = os.getenv("DB_URL", f"sqlite:///{db_path.absolute()}")
    
    all_results = {}
    all_metrics = {}
    
    # ========================================================================
    # 1. QWEN BASE MODEL (Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("1. EVALUATING QWEN 0.5 BASE MODEL (Baseline)")
    print("="*80)
    try:
        base_results, base_metrics = run_model_evaluation(
            "ollama/qwen-0.5b-base",
            "QWEN 0.5 BASE MODEL",
            test_cases,
            db_url,
            output_dir,
            "Make sure qwen_base_server.py is running on port 11439!"
        )
        all_results['base'] = base_results
        all_metrics['base'] = base_metrics
    except Exception as e:
        print(f"\n⚠️  ERROR during base model evaluation: {e}")
        import traceback
        traceback.print_exc()
        all_results['base'] = None
        all_metrics['base'] = None
    
    # ========================================================================
    # 2. QWEN FINETUNED MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("2. EVALUATING QWEN 0.5 FINETUNED MODEL")
    print("="*80)
    try:
        finetuned_results, finetuned_metrics = run_model_evaluation(
            "ollama/qwen-0.5b-spider",
            "QWEN 0.5 FINETUNED MODEL",
            test_cases,
            db_url,
            output_dir,
            "Make sure qwen_server.py is running on port 11438!"
        )
        all_results['finetuned'] = finetuned_results
        all_metrics['finetuned'] = finetuned_metrics
    except Exception as e:
        print(f"\n⚠️  ERROR during finetuned model evaluation: {e}")
        import traceback
        traceback.print_exc()
        all_results['finetuned'] = None
        all_metrics['finetuned'] = None
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON - QWEN BASE vs FINETUNED")
    print("="*80)
    
    if all_metrics['base'] and all_metrics['finetuned']:
        print("\n📊 Execution Accuracy (EX) Comparison:")
        print(f"{'Model':<25} {'With RAG':<15} {'Without RAG':<15} {'Improvement':<15}")
        print("-" * 70)
        
        base_m = all_metrics['base']
        finetuned_m = all_metrics['finetuned']
        
        base_improvement = base_m['with_rag_ex_count'] - base_m['without_rag_ex_count']
        finetuned_improvement = finetuned_m['with_rag_ex_count'] - finetuned_m['without_rag_ex_count']
        
        print(f"{'Qwen 0.5 Base':<25} {base_m['with_rag_ex_pct']:>6.1f}% ({base_m['with_rag_ex_count']:>2}/{base_m['total']:<2})  {base_m['without_rag_ex_pct']:>6.1f}% ({base_m['without_rag_ex_count']:>2}/{base_m['total']:<2})  {base_improvement:>+6.1f}%")
        print(f"{'Qwen 0.5 Finetuned':<25} {finetuned_m['with_rag_ex_pct']:>6.1f}% ({finetuned_m['with_rag_ex_count']:>2}/{finetuned_m['total']:<2})  {finetuned_m['without_rag_ex_pct']:>6.1f}% ({finetuned_m['without_rag_ex_count']:>2}/{finetuned_m['total']:<2})  {finetuned_improvement:>+6.1f}%")
        
        # Calculate improvement from fine-tuning
        ex_improvement_with_rag = finetuned_m['with_rag_ex_count'] - base_m['with_rag_ex_count']
        ex_improvement_without_rag = finetuned_m['without_rag_ex_count'] - base_m['without_rag_ex_count']
        ex_improvement_pct_with = (ex_improvement_with_rag / base_m['total']) * 100
        ex_improvement_pct_without = (ex_improvement_without_rag / base_m['total']) * 100
        
        print(f"\n🎯 Fine-tuning Impact:")
        print(f"   With RAG:    {ex_improvement_with_rag:+d} ({ex_improvement_pct_with:+.1f}%)")
        print(f"   Without RAG:  {ex_improvement_without_rag:+d} ({ex_improvement_pct_without:+.1f}%)")
        
        # Latency comparison
        print("\n⏱️  Average Latency Comparison:")
        print(f"{'Model':<25} {'With RAG':<15} {'Without RAG':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Calculate latencies for base
        base_latencies_with = []
        base_latencies_without = []
        for technique in TECHNIQUES:
            if all_results['base'] and technique in all_results['base']:
                df = all_results['base'][technique]
                for lat_str in df['With_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            base_latencies_with.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
                for lat_str in df['Without_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            base_latencies_without.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
        
        # Calculate latencies for finetuned
        finetuned_latencies_with = []
        finetuned_latencies_without = []
        for technique in TECHNIQUES:
            if all_results['finetuned'] and technique in all_results['finetuned']:
                df = all_results['finetuned'][technique]
                for lat_str in df['With_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            finetuned_latencies_with.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
                for lat_str in df['Without_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            finetuned_latencies_without.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
        
        base_avg_with = sum(base_latencies_with) / len(base_latencies_with) if base_latencies_with else 0
        base_avg_without = sum(base_latencies_without) / len(base_latencies_without) if base_latencies_without else 0
        finetuned_avg_with = sum(finetuned_latencies_with) / len(finetuned_latencies_with) if finetuned_latencies_with else 0
        finetuned_avg_without = sum(finetuned_latencies_without) / len(finetuned_latencies_without) if finetuned_latencies_without else 0
        
        print(f"{'Qwen 0.5 Base':<25} {base_avg_with:>8.2f}s      {base_avg_without:>8.2f}s      {base_avg_with - base_avg_without:>+8.2f}s")
        print(f"{'Qwen 0.5 Finetuned':<25} {finetuned_avg_with:>8.2f}s      {finetuned_avg_without:>8.2f}s      {finetuned_avg_with - finetuned_avg_without:>+8.2f}s")
        
        latency_diff_with = finetuned_avg_with - base_avg_with
        latency_diff_without = finetuned_avg_without - base_avg_without
        print(f"\n⏱️  Fine-tuning Latency Impact:")
        print(f"   With RAG:    {latency_diff_with:+.2f}s")
        print(f"   Without RAG: {latency_diff_without:+.2f}s")
        
        print("\n✅ Success Rate Comparison:")
        print(f"{'Model':<25} {'With RAG':<15} {'Without RAG':<15}")
        print("-" * 55)
        print(f"{'Qwen 0.5 Base':<25} {base_m['with_rag_success_pct']:>6.1f}% ({base_m['with_rag_success_count']:>2}/{base_m['total']:<2})  {base_m['without_rag_success_pct']:>6.1f}% ({base_m['without_rag_success_count']:>2}/{base_m['total']:<2})")
        print(f"{'Qwen 0.5 Finetuned':<25} {finetuned_m['with_rag_success_pct']:>6.1f}% ({finetuned_m['with_rag_success_count']:>2}/{finetuned_m['total']:<2})  {finetuned_m['without_rag_success_pct']:>6.1f}% ({finetuned_m['without_rag_success_count']:>2}/{finetuned_m['total']:<2})")
        
    else:
        print("\n⚠️  Cannot generate comparison - one or both models failed to evaluate")
        if not all_metrics['base']:
            print("   ❌ Base model evaluation failed")
        if not all_metrics['finetuned']:
            print("   ❌ Finetuned model evaluation failed")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Test cases saved to: {test_cases_file}")


if __name__ == "__main__":
    main()

