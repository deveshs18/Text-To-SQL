"""
Compare GPT-4o-mini vs Qwen 0.5 Finetuned vs Arctic Base.
All models use standardized prompts and SQL corrections for fair comparison.
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
    """Main function to compare GPT vs Qwen vs Arctic."""
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print("Comparing: GPT-4o-mini vs Qwen 0.5 Finetuned vs Arctic Base")
    print("All models use standardized prompts and SQL corrections")
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"Techniques: {', '.join(TECHNIQUES)}")
    print(f"RAG Settings: With RAG & Without RAG")
    print("="*80)
    print("\n📝 IMPORTANT NOTES:")
    print("   1. Make sure qwen_server.py is running on port 11438 (Qwen finetuned)")
    print("   2. Make sure arctic_base_server.py is running on port 11437 (Arctic)")
    print("   3. GPT-4o-mini uses OpenAI API (no server needed)")
    print("   4. Each query has timeout protection (240s per case)")
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
    
    test_cases_file = output_dir / f"all_models_comparison_{NUM_TEST_CASES}_test_cases.json"
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
    # 1. QWEN 0.5 FINETUNED MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("1. EVALUATING QWEN 0.5 FINETUNED MODEL")
    print("="*80)
    try:
        qwen_results, qwen_metrics = run_model_evaluation(
            "ollama/qwen-0.5b-spider",
            "QWEN 0.5 FINETUNED MODEL",
            test_cases,
            db_url,
            output_dir,
            "Make sure qwen_server.py is running on port 11438!"
        )
        all_results['qwen'] = qwen_results
        all_metrics['qwen'] = qwen_metrics
    except Exception as e:
        print(f"\n⚠️  ERROR during Qwen evaluation: {e}")
        import traceback
        traceback.print_exc()
        all_results['qwen'] = None
        all_metrics['qwen'] = None
    
    # ========================================================================
    # 2. ARCTIC BASE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("2. EVALUATING ARCTIC BASE MODEL")
    print("="*80)
    try:
        arctic_results, arctic_metrics = run_model_evaluation(
            "ollama/arctic-base",
            "ARCTIC BASE MODEL",
            test_cases,
            db_url,
            output_dir,
            "Make sure arctic_base_server.py is running on port 11437!"
        )
        all_results['arctic'] = arctic_results
        all_metrics['arctic'] = arctic_metrics
    except Exception as e:
        print(f"\n⚠️  ERROR during Arctic evaluation: {e}")
        import traceback
        traceback.print_exc()
        all_results['arctic'] = None
        all_metrics['arctic'] = None
    
    # ========================================================================
    # 3. GPT-4O-MINI MODEL
    # ========================================================================
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "sk-your-key-here":
        print("\n" + "="*80)
        print("⚠️  WARNING: OPENAI_API_KEY not found or invalid in .env")
        print("Skipping GPT-4o-mini evaluation")
        print("="*80)
        all_results['gpt'] = None
        all_metrics['gpt'] = None
    else:
        print("\n" + "="*80)
        print("3. EVALUATING GPT-4O-MINI MODEL")
        print("="*80)
        try:
            gpt_results, gpt_metrics = run_model_evaluation(
                "openai/gpt-4o-mini",
                "GPT-4O-MINI",
                test_cases,
                db_url,
                output_dir
            )
            all_results['gpt'] = gpt_results
            all_metrics['gpt'] = gpt_metrics
        except Exception as e:
            print(f"\n⚠️  ERROR during GPT evaluation: {e}")
            import traceback
            traceback.print_exc()
            all_results['gpt'] = None
            all_metrics['gpt'] = None
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL MODELS")
    print("="*80)
    
    if all_metrics.get('qwen') and all_metrics.get('arctic') and all_metrics.get('gpt'):
        print("\n📊 EXECUTION ACCURACY (EX) & SUCCESS RATE COMPARISON (With RAG):")
        print(f"{'Model':<25} {'EX':<12} {'Success':<12} {'EX Count':<12} {'Success Count':<15} {'Latency':<12}")
        print("-" * 95)
        
        for model_key, model_name in [('qwen', 'Qwen 0.5 Finetuned'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
            if all_metrics[model_key]:
                m = all_metrics[model_key]
                # Calculate latency
                latencies = []
                if all_results[model_key]:
                    for technique in TECHNIQUES:
                        df = all_results[model_key][technique]
                        for lat_str in df['With_RAG_Latency']:
                            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                                try:
                                    latencies.append(float(str(lat_str).replace('s', '')))
                                except:
                                    pass
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                
                print(f"{model_name:<25} {m['with_rag_ex_pct']:>6.1f}%     {m['with_rag_success_pct']:>6.1f}%     {m['with_rag_ex_count']:>2}/{m['total']:<8} {m['with_rag_success_count']:>2}/{m['total']:<11} {avg_latency:>8.2f}s")
        
        print("\n📊 EXECUTION ACCURACY (EX) & SUCCESS RATE COMPARISON (Without RAG):")
        print(f"{'Model':<25} {'EX':<12} {'Success':<12} {'EX Count':<12} {'Success Count':<15} {'Latency':<12}")
        print("-" * 95)
        
        for model_key, model_name in [('qwen', 'Qwen 0.5 Finetuned'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
            if all_metrics[model_key]:
                m = all_metrics[model_key]
                # Calculate latency
                latencies = []
                if all_results[model_key]:
                    for technique in TECHNIQUES:
                        df = all_results[model_key][technique]
                        for lat_str in df['Without_RAG_Latency']:
                            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                                try:
                                    latencies.append(float(str(lat_str).replace('s', '')))
                                except:
                                    pass
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                
                print(f"{model_name:<25} {m['without_rag_ex_pct']:>6.1f}%     {m['without_rag_success_pct']:>6.1f}%     {m['without_rag_ex_count']:>2}/{m['total']:<8} {m['without_rag_success_count']:>2}/{m['total']:<11} {avg_latency:>8.2f}s")
        
        print("\n🎯 RAG IMPACT - EX & SUCCESS RATE IMPROVEMENT:")
        print(f"{'Model':<25} {'EX Improvement':<18} {'Success Improvement':<20}")
        print("-" * 65)
        
        for model_key, model_name in [('qwen', 'Qwen 0.5 Finetuned'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
            if all_metrics[model_key]:
                m = all_metrics[model_key]
                ex_improvement = m['with_rag_ex_count'] - m['without_rag_ex_count']
                success_improvement = m['with_rag_success_count'] - m['without_rag_success_count']
                ex_improvement_pct = ((ex_improvement/m['total'])*100)
                success_improvement_pct = ((success_improvement/m['total'])*100)
                print(f"{model_name:<25} {ex_improvement:>+4} ({ex_improvement_pct:>+6.1f}%)      {success_improvement:>+4} ({success_improvement_pct:>+6.1f}%)")
        
        print("\n🏆 MODEL RANKING (With RAG):")
        print("Ranked by Execution Accuracy (EX):")
        models_ranked = []
        for model_key, model_name in [('qwen', 'Qwen 0.5 Finetuned'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
            if all_metrics[model_key]:
                m = all_metrics[model_key]
                models_ranked.append((m['with_rag_ex_pct'], model_name, m['with_rag_success_pct']))
        models_ranked.sort(reverse=True)
        for rank, (ex_pct, name, success_pct) in enumerate(models_ranked, 1):
            print(f"  {rank}. {name:<25} EX: {ex_pct:>6.1f}%  Success: {success_pct:>6.1f}%")
        
    else:
        print("\n⚠️  Cannot generate full comparison - one or more models failed to evaluate")
        if not all_metrics.get('qwen'):
            print("   ❌ Qwen model evaluation failed")
        if not all_metrics.get('arctic'):
            print("   ❌ Arctic model evaluation failed")
        if not all_metrics.get('gpt'):
            print("   ❌ GPT model evaluation failed")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Test cases saved to: {test_cases_file}")


if __name__ == "__main__":
    main()

