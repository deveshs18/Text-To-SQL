"""
Evaluate GPT-4o-mini model only.
Runs evaluation on GPT-4o-mini with all prompting techniques and saves results.
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


def main():
    """Main function to evaluate GPT-4o-mini."""
    print("="*80)
    print("GPT-4O-MINI EVALUATION")
    print("="*80)
    print("Model: GPT-4o-mini (OpenAI API)")
    print("All models use model-specific optimized prompts")
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"Techniques: {', '.join(TECHNIQUES)}")
    print(f"RAG Settings: With RAG & Without RAG")
    print("="*80)
    print("\n📝 IMPORTANT NOTES:")
    print("   1. GPT-4o-mini uses OpenAI API (no server needed)")
    print("   2. Make sure OPENAI_API_KEY is set in .env file")
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
    
    test_cases_file = output_dir / f"gpt_evaluation_{NUM_TEST_CASES}_test_cases.json"
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
    
    # ========================================================================
    # EVALUATE GPT-4O-MINI
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING GPT-4O-MINI")
    print("="*80)
    
    try:
        # Run evaluation
        print(f"\nModel: openai/gpt-4o-mini")
        print(f"Test Cases: {len(test_cases)}")
        print(f"Techniques: {', '.join(TECHNIQUES)}")
        print(f"RAG Settings: With RAG & Without RAG")
        print("="*80 + "\n")
        
        results = run_evaluation_for_model(
            "openai/gpt-4o-mini",
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
        print("GPT-4O-MINI - SUMMARY")
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
        
        print("="*80)
        print("\n✅ EVALUATION COMPLETE!")
        print("="*80)
        print(f"All results saved to: {output_dir}")
        print(f"Test cases saved to: {test_cases_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR during GPT evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

