"""
Sequential evaluation script for 3 models (Qwen, Arctic, GPT-4o-mini).
Runs each model separately with 35 test cases, all 4 prompts, with/without RAG.
Focuses on EX (Execution Accuracy) and latency.
"""
import os
import sys
import json
import gc
import torch
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
load_test_cases = compare_rag.load_test_cases

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


def clear_memory():
    """Clear GPU and CPU memory."""
    print("\n" + "="*80)
    print("CLEARING MEMORY (Unloading Previous Model)...")
    print("="*80)
    
    # Clear Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU cache cleared (model unloaded from GPU)")
    
    print("✅ Memory cleared - Ready to load next model\n")


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
    
    # Print summary focusing on EX and latency
    print("\n" + "="*80)
    print(f"{model_display_name} - SUMMARY (EX & LATENCY FOCUS)")
    print("="*80)
    
    print(f"\n📊 Execution Accuracy (EX):")
    print(f"   With RAG:    {metrics['with_rag_ex_count']}/{metrics['total']} ({metrics['with_rag_ex_pct']:.1f}%)")
    print(f"   Without RAG: {metrics['without_rag_ex_count']}/{metrics['total']} ({metrics['without_rag_ex_pct']:.1f}%)")
    print(f"   Improvement: {metrics['with_rag_ex_count'] - metrics['without_rag_ex_count']:+d} ({((metrics['with_rag_ex_count'] - metrics['without_rag_ex_count'])/metrics['total']*100):+.1f}%)")
    
    # Calculate average latency from results
    with_rag_latencies = []
    without_rag_latencies = []
    
    for technique in TECHNIQUES:
        df = results[technique]
        # Extract latency from string format "X.XXs" or "SKIPPED"
        for lat_str in df['With_RAG_Latency']:
            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                try:
                    with_rag_latencies.append(float(lat_str.replace('s', '')))
                except:
                    pass
        for lat_str in df['Without_RAG_Latency']:
            if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                try:
                    without_rag_latencies.append(float(lat_str.replace('s', '')))
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
    """Main function to run sequential evaluation."""
    print("="*80)
    print("SEQUENTIAL MODEL EVALUATION")
    print("="*80)
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"Techniques: {', '.join(TECHNIQUES)}")
    print(f"RAG Settings: With RAG & Without RAG")
    print("="*80)
    print("\n📝 IMPORTANT NOTES:")
    print("   ✅ Queries will NOT hang - timeout protection is enabled (240s per case)")
    print("   ✅ Memory is automatically cleared between models")
    print("   ✅ Models are automatically loaded/unloaded sequentially")
    print("   ✅ Each query has individual timeout protection")
    print("="*80)
    
    # Load test cases
    print("\nLoading test cases...")
    all_test_cases = load_all_100_test_cases()
    
    # Take only first 35 test cases
    if len(all_test_cases) >= NUM_TEST_CASES:
        test_cases = all_test_cases[:NUM_TEST_CASES]
    else:
        print(f"Warning: Only {len(all_test_cases)} test cases available. Using all available.")
        test_cases = all_test_cases
    
    print(f"✅ Loaded {len(test_cases)} test cases\n")
    
    # Save test cases
    output_dir = script_dir / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases_file = output_dir / f"all_{NUM_TEST_CASES}_test_cases.json"
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
    # 1. QWEN MODEL (Load from saved results)
    # ========================================================================
    print("\n" + "="*80)
    print("📂 LOADING SAVED QWEN RESULTS (Skipping evaluation)")
    print("="*80)
    try:
        # Load saved Qwen results instead of running evaluation
        from pathlib import Path
        qwen_results = {}
        for technique in TECHNIQUES:
            csv_path = output_dir / f"ollama_qwen-0.5b-spider_{technique}_results.csv"
            if csv_path.exists():
                qwen_results[technique] = pd.read_csv(csv_path)
                print(f"✓ Loaded {technique} results: {len(qwen_results[technique])} rows")
            else:
                print(f"⚠️  Warning: {csv_path} not found")
        
        if qwen_results:
            all_results_df = pd.concat([qwen_results[t] for t in TECHNIQUES], ignore_index=True)
            qwen_metrics = calculate_overall_metrics(all_results_df)
            all_results['qwen'] = qwen_results
            all_metrics['qwen'] = qwen_metrics
            print(f"\n✅ Qwen results loaded successfully!")
            print(f"   Total queries: {qwen_metrics['total']}")
            print(f"   EX (With RAG): {qwen_metrics['with_rag_ex_count']}/{qwen_metrics['total']} ({qwen_metrics['with_rag_ex_pct']:.1f}%)")
        else:
            print("⚠️  No Qwen results found. Skipping Qwen.")
            all_results['qwen'] = None
            all_metrics['qwen'] = None
    except Exception as e:
        print(f"\n⚠️  ERROR loading Qwen results: {e}")
        print("Continuing to Arctic model...")
        all_results['qwen'] = None
        all_metrics['qwen'] = None
    
    # AUTOMATIC: Clear memory and prepare for next model
    print("\n" + "="*80)
    print("🔄 AUTOMATIC MODEL SWITCHING: Qwen Complete → Clearing Memory → Loading Arctic")
    print("="*80)
    clear_memory()
    
    # ========================================================================
    # 2. ARCTIC MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("🔄 AUTOMATIC MODEL SWITCHING: Starting Arctic Model")
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
        print("Continuing to GPT model...")
        all_results['arctic'] = None
        all_metrics['arctic'] = None
    
    # AUTOMATIC: Clear memory and prepare for next model
    print("\n" + "="*80)
    print("🔄 AUTOMATIC MODEL SWITCHING: Arctic Complete → Clearing Memory → Loading GPT")
    print("="*80)
    clear_memory()
    
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
        print("🔄 AUTOMATIC MODEL SWITCHING: Starting GPT-4o-mini Model")
        print("="*80)
        gpt_results, gpt_metrics = run_model_evaluation(
            "openai/gpt-4o-mini",
            "GPT-4O-MINI",
            test_cases,
            db_url,
            output_dir
        )
        all_results['gpt'] = gpt_results
        all_metrics['gpt'] = gpt_metrics
        print("\n" + "="*80)
        print("🔄 AUTOMATIC MODEL SWITCHING: GPT Complete → All Models Evaluated")
        print("="*80)
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL MODELS")
    print("="*80)
    
    print("\n📊 Execution Accuracy (EX) Comparison:")
    print(f"{'Model':<20} {'With RAG':<15} {'Without RAG':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for model_key, model_name in [('qwen', 'Qwen-0.5B'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
        if all_metrics[model_key]:
            m = all_metrics[model_key]
            improvement = m['with_rag_ex_count'] - m['without_rag_ex_count']
            print(f"{model_name:<20} {m['with_rag_ex_pct']:>6.1f}% ({m['with_rag_ex_count']:>2}/{m['total']:<2})  {m['without_rag_ex_pct']:>6.1f}% ({m['without_rag_ex_count']:>2}/{m['total']:<2})  {improvement:>+6.1f}%")
    
    print("\n⏱️  Average Latency Comparison:")
    print(f"{'Model':<20} {'With RAG':<15} {'Without RAG':<15} {'Difference':<15}")
    print("-" * 65)
    
    for model_key, model_name in [('qwen', 'Qwen-0.5B'), ('arctic', 'Arctic Base'), ('gpt', 'GPT-4o-mini')]:
        if all_results[model_key]:
            # Calculate average latency
            latencies_with = []
            latencies_without = []
            for technique in TECHNIQUES:
                df = all_results[model_key][technique]
                # Extract latency from string format "X.XXs" or "SKIPPED"
                for lat_str in df['With_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            latencies_with.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
                for lat_str in df['Without_RAG_Latency']:
                    if lat_str != 'SKIPPED' and lat_str != 'TIMEOUT':
                        try:
                            latencies_without.append(float(str(lat_str).replace('s', '')))
                        except:
                            pass
            
            avg_with = sum(latencies_with) / len(latencies_with) if latencies_with else 0
            avg_without = sum(latencies_without) / len(latencies_without) if latencies_without else 0
            diff = avg_with - avg_without
            
            print(f"{model_name:<20} {avg_with:>8.2f}s      {avg_without:>8.2f}s      {diff:>+8.2f}s")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Test cases saved to: {test_cases_file}")


if __name__ == "__main__":
    main()

