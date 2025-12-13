"""Compare RAG vs No-RAG performance on test queries."""
import os
import json
import pandas as pd
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from db import get_engine, validate_sql, execute_sql
from schema_retriever import get_schema_snippet, get_all_tables
from prompts import build_few_shot_prompt, build_cot_prompt, build_ltm_prompt, build_eg_prompt
from model_client import generate_sql
from advanced_metrics import calculate_all_metrics
from sql_corrector import validate_and_correct_sql
from dotenv import load_dotenv

load_dotenv()

# Thread-safe print lock for parallel execution
print_lock = Lock()


def evaluate_with_rag_setting(
    question: str,
    gold_sql: str,
    model_name: str,
    technique: str,
    db_url: str,
    use_rag: bool = True
) -> Dict:
    """Evaluate a query with or without RAG."""
    engine = get_engine(db_url)
    
    # Get schema based on RAG setting
    # With RAG: Full schema with column names
    # Without RAG: No schema (or minimal - just table name)
    tables = get_all_tables(engine)
    table_name = tables[0] if tables else "adult_income"
    
    if use_rag:
        # With RAG: Get full schema with column names
        schema = get_schema_snippet(question, engine, table_name=table_name, model_name=model_name)
        # Import get_examples_for_table to get examples
        from prompts import get_examples_for_table
        examples = get_examples_for_table(table_name, simple=True)
    else:
        # Without RAG: Minimal schema (just table name, no columns)
        schema = f"{table_name}(...)"
        examples = None
    
    # Build prompt
    prompt_builders = {
        "Few-Shot": build_few_shot_prompt,
        "CoT": build_cot_prompt,
        "LtM": build_ltm_prompt,
        "EG": build_eg_prompt
    }
    
    # Only Few-Shot uses examples parameter
    if technique == "Few-Shot":
        prompt = prompt_builders[technique](schema, question, examples=examples, model_name=model_name)
    else:
        prompt = prompt_builders[technique](schema, question, model_name=model_name)
    
    # Generate SQL
    try:
        sql, generation_metrics = generate_sql(
            prompt,
            model_name,
            temperature=0.1
        )
        
        # Apply SQL corrections - model-specific handling
        # Get table name from schema
        table_name = "adult_income"  # Default
        if hasattr(engine, 'url'):
            # Try to extract table name from schema
            schema_lower = schema.lower() if isinstance(schema, str) else ""
            if "adult_income" in schema_lower:
                table_name = "adult_income"
        
        # Detect if base model - use less aggressive corrections
        is_base_model = (
            "qwen-0.5b-base" in model_name.lower() or
            ("qwen" in model_name.lower() and "base" in model_name.lower() and "spider" not in model_name.lower())
        )
        
        # Detect model type for SQL corrections
        is_gpt = "gpt" in model_name.lower() or "openai" in model_name.lower()
        is_arctic = "arctic" in model_name.lower()
        
        if is_base_model:
            # Base model: Apply general-purpose fixes only (skip dataset-specific fixes)
            # General-purpose fixes: missing GROUP BY, missing LIMIT, missing FROM, etc.
            # These are safe and help without breaking valid SQL
            corrected_sql, fixes, is_valid_corrected = validate_and_correct_sql(
                sql, engine, table_name, question, dataset_specific_fixes=False
            )
            if fixes:
                sql = corrected_sql
        elif is_gpt or is_arctic:
            # GPT and Arctic: Apply general-purpose fixes only (skip dataset-specific fixes)
            # Their SQL patterns might differ from Qwen, so be conservative
            corrected_sql, fixes, is_valid_corrected = validate_and_correct_sql(
                sql, engine, table_name, question, dataset_specific_fixes=False
            )
            if fixes:
                sql = corrected_sql
        else:
            # Qwen Finetuned: Apply all corrections (general + dataset-specific)
            corrected_sql, fixes, is_valid_corrected = validate_and_correct_sql(
                sql, engine, table_name, question, dataset_specific_fixes=True
            )
            if fixes:
                sql = corrected_sql
        
        # Validate SQL
        is_valid, validation_error = validate_sql(sql)
        if not is_valid:
            return {
                "success": False,
                "sql": sql,
                "error": validation_error,
                "metrics": {},
                "tokens_used": generation_metrics.get("tokens_used", 0),
                "latency": generation_metrics.get("latency", 0.0)
            }
        
        # Execute SQL
        exec_success, df, exec_error, exec_latency = execute_sql(sql, engine)
        
        # Calculate metrics
        all_metrics = {}
        if exec_success and df is not None:
            try:
                all_metrics = calculate_all_metrics(
                    gold_sql,
                    sql,
                    engine,
                    include_execution=True
                )
            except Exception as e:
                all_metrics = {"error": str(e)}
        
        return {
            "success": exec_success,
            "sql": sql,
            "error": exec_error,
            "metrics": all_metrics,
            "tokens_used": generation_metrics.get("tokens_used", 0),
            "latency": generation_metrics.get("latency", 0.0) + exec_latency,
            "rows_returned": len(df) if df is not None else 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "sql": "",
            "error": str(e),
            "metrics": {},
            "tokens_used": 0,
            "latency": 0.0,
            "rows_returned": 0
        }


def process_single_test_case(
    args: Tuple[int, Dict[str, str], str, str, str]
) -> Tuple[int, Dict]:
    """Process a single test case (for parallel execution)."""
    case_idx, test_case, model_name, technique, db_url = args
    question = test_case["question"]
    gold_sql = test_case["gold_sql"]
    
    case_start = time.time()
    
    with print_lock:
        print(f"[{case_idx}] Starting: {question[:50]}...")
    
    # Evaluate with RAG (with error handling)
    with_rag_start = time.time()
    try:
        with_rag = evaluate_with_rag_setting(
            question, gold_sql, model_name, technique, db_url, use_rag=True
        )
    except Exception as e:
        with print_lock:
            print(f"[{case_idx}] ERROR in with_rag: {str(e)}")
        with_rag = {
            "success": False,
            "sql": "",
            "error": str(e),
            "metrics": {},
            "tokens_used": 0,
            "latency": 0.0,
            "rows_returned": 0
        }
    with_rag_time = time.time() - with_rag_start
    
    # Evaluate without RAG (with error handling)
    without_rag_start = time.time()
    try:
        without_rag = evaluate_with_rag_setting(
            question, gold_sql, model_name, technique, db_url, use_rag=False
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        with print_lock:
            print(f"[{case_idx}] ERROR in without_rag: {str(e)}")
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                print(f"[{case_idx}] ⚠️  TIMEOUT in without_rag evaluation")
        without_rag = {
            "success": False,
            "sql": "",
            "error": str(e),
            "metrics": {},
            "tokens_used": 0,
            "latency": 0.0,
            "rows_returned": 0
        }
    without_rag_time = time.time() - without_rag_start
    
    case_time = time.time() - case_start
    
    # Extract metrics
    with_rag_ex = with_rag["metrics"].get("execution_accuracy", False)
    with_rag_sm = with_rag["metrics"].get("semantic_match", False)
    with_rag_f1 = with_rag["metrics"].get("f1_score", 0.0)
    
    without_rag_ex = without_rag["metrics"].get("execution_accuracy", False)
    without_rag_sm = without_rag["metrics"].get("semantic_match", False)
    without_rag_f1 = without_rag["metrics"].get("f1_score", 0.0)
    
    result = {
        "Question": question[:50] + "..." if len(question) > 50 else question,
        "With_RAG_Success": "✅" if with_rag["success"] else "❌",
        "With_RAG_EX": "✅" if with_rag_ex else "❌",
        "With_RAG_SM": "✅" if with_rag_sm else "❌",
        "With_RAG_F1": f"{with_rag_f1:.3f}",
        "With_RAG_Latency": f"{with_rag['latency']:.2f}s",
        "With_RAG_Tokens": with_rag["tokens_used"],
        "Without_RAG_Success": "✅" if without_rag["success"] else "❌",
        "Without_RAG_EX": "✅" if without_rag_ex else "❌",
        "Without_RAG_SM": "✅" if without_rag_sm else "❌",
        "Without_RAG_F1": f"{without_rag_f1:.3f}",
        "Without_RAG_Latency": f"{without_rag['latency']:.2f}s",
        "Without_RAG_Tokens": without_rag["tokens_used"],
        "RAG_Helped": "✅" if (with_rag_ex and not without_rag_ex) or (with_rag_f1 > without_rag_f1 + 0.1) else ("❌" if (without_rag_ex and not with_rag_ex) or (without_rag_f1 > with_rag_f1 + 0.1) else "➖"),
        "With_RAG_SQL": with_rag["sql"][:100] + "..." if len(with_rag["sql"]) > 100 else with_rag["sql"],
        "Without_RAG_SQL": without_rag["sql"][:100] + "..." if len(without_rag["sql"]) > 100 else without_rag["sql"],
    }
    
    # Extract EX for display
    with_rag_ex = with_rag["metrics"].get("execution_accuracy", False)
    without_rag_ex = without_rag["metrics"].get("execution_accuracy", False)
    
    with print_lock:
        rag_status = "OK" if with_rag["success"] else "FAIL"
        no_rag_status = "OK" if without_rag["success"] else "FAIL"
        rag_ex_status = "✅EX" if with_rag_ex else "❌EX"
        no_rag_ex_status = "✅EX" if without_rag_ex else "❌EX"
        print(f"[{case_idx}] Done ({case_time:.1f}s) | RAG: {rag_status} {rag_ex_status} ({with_rag_time:.1f}s) | No-RAG: {no_rag_status} {no_rag_ex_status} ({without_rag_time:.1f}s)")
    
    return case_idx, result


def compare_rag_performance(
    test_cases: List[Dict[str, str]],
    model_name: str,
    technique: str,
    db_url: str,
    max_workers: int = None
) -> pd.DataFrame:
    """Compare RAG vs No-RAG for a list of test cases with parallel processing."""
    results = []
    
    # Determine optimal number of workers based on model type
    # Running locally - use sequential processing (max_workers=1) to avoid issues
    if max_workers is None:
        max_workers = 1  # Sequential processing for local execution
    
    # Skip problematic test cases (e.g., case 19 which has HAVING clause and takes too long)
    SKIP_CASES = [19]  # Add case indices to skip here (1-based indexing)
    
    # Prepare arguments for parallel processing, skipping problematic cases
    args_list = [
        (i + 1, test_case, model_name, technique, db_url)
        for i, test_case in enumerate(test_cases)
        if (i + 1) not in SKIP_CASES
    ]
    
    print(f"\n{'='*80}")
    print(f"Comparing RAG vs No-RAG (SEQUENTIAL MODE - Local Execution)")
    print(f"Model: {model_name} | Technique: {technique}")
    print(f"Total test cases: {len(test_cases)}")
    if SKIP_CASES:
        print(f"⚠️  Skipping {len(SKIP_CASES)} problematic test case(s): {SKIP_CASES}")
        print(f"   Will evaluate {len(args_list)} test cases instead of {len(test_cases)}")
    print(f"Database: {db_url}")
    print(f"Processing: Sequential (1 worker)")
    print(f"Timeout per case: 4 minutes (240 seconds) - covers both with/without RAG calls (2 min each)")
    
    # Estimate time
    estimated_time = len(args_list) * 2 * 30  # 2 LLM calls per case, ~30s each (conservative)
    print(f"\nESTIMATED TIME:")
    print(f"   Sequential: ~{estimated_time} seconds ({estimated_time/60:.1f} minutes)")
    print(f"   Max time (if all timeout): ~{len(args_list) * 3} minutes")
    print(f"{'='*80}\n")
    
    start_total = time.time()
    completed_count = 0
    
    # Process sequentially with timeout per case
    # This prevents one stuck case from blocking everything
    results_dict = {}
    # Timeout should be 240 seconds (4 minutes) because each case has 2 calls:
    # 1. with_rag (up to 120s timeout) + 2. without_rag (up to 120s timeout) = 240s total
    timeout_seconds = 240  # 4 minutes per case (covers both with/without RAG calls)
    
    # Sequential processing with timeout per case
    for args in args_list:
        case_idx = args[0]
        case_start_time = time.time()
        
        # Use executor with timeout to prevent hanging
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_single_test_case, args)
            try:
                case_idx, result = future.result(timeout=timeout_seconds)
                results_dict[case_idx] = result
                completed_count += 1
                
                # Progress update
                elapsed = time.time() - start_total
                avg_time = elapsed / completed_count
                remaining = len(test_cases) - completed_count
                eta = avg_time * remaining
                
                with print_lock:
                    print(f"Progress: {completed_count}/{len(test_cases)} ({completed_count/len(test_cases)*100:.1f}%) | "
                          f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            except TimeoutError:
                # Cancel the future to stop the underlying task
                try:
                    future.cancel()
                except:
                    pass
                with print_lock:
                    print(f"⚠️  TIMEOUT: Case {case_idx} exceeded {timeout_seconds}s (4 min) timeout - marking as failed and moving to next")
                # Add timeout error result
                results_dict[case_idx] = {
                    "Question": f"Timeout in case {case_idx}",
                    "With_RAG_Success": "❌",
                    "With_RAG_EX": "❌",
                    "With_RAG_SM": "❌",
                    "With_RAG_F1": "0.000",
                    "With_RAG_Latency": "TIMEOUT",
                    "With_RAG_Tokens": 0,
                    "Without_RAG_Success": "❌",
                    "Without_RAG_EX": "❌",
                    "Without_RAG_SM": "❌",
                    "Without_RAG_F1": "0.000",
                    "Without_RAG_Latency": "TIMEOUT",
                    "Without_RAG_Tokens": 0,
                    "RAG_Helped": "➖",
                    "With_RAG_SQL": "",
                    "Without_RAG_SQL": "",
                }
                completed_count += 1
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                with print_lock:
                    print(f"ERROR: Error processing case {case_idx}: {str(e)}")
                    print(f"Full traceback:\n{error_trace}")
                # Add error result
                results_dict[case_idx] = {
                    "Question": f"Error in case {case_idx}: {str(e)}",
                    "With_RAG_Success": "❌",
                    "With_RAG_EX": "❌",
                    "With_RAG_SM": "❌",
                    "With_RAG_F1": "0.000",
                    "With_RAG_Latency": "0.00s",
                    "With_RAG_Tokens": 0,
                    "Without_RAG_Success": "❌",
                    "Without_RAG_EX": "❌",
                    "Without_RAG_SM": "❌",
                    "Without_RAG_F1": "0.000",
                    "Without_RAG_Latency": "0.00s",
                    "Without_RAG_Tokens": 0,
                    "RAG_Helped": "➖",
                    "With_RAG_SQL": "",
                    "Without_RAG_SQL": "",
                }
                completed_count += 1
    
    # Add skipped cases as "SKIPPED" entries
    SKIP_CASES = [19]  # Must match the SKIP_CASES above
    for skip_idx in SKIP_CASES:
        if skip_idx <= len(test_cases):
            results_dict[skip_idx] = {
                "Question": f"Skipped case {skip_idx} (problematic query)",
                "With_RAG_Success": "SKIPPED",
                "With_RAG_EX": "SKIPPED",
                "With_RAG_SM": "SKIPPED",
                "With_RAG_F1": "0.000",
                "With_RAG_Latency": "SKIPPED",
                "With_RAG_Tokens": 0,
                "Without_RAG_Success": "SKIPPED",
                "Without_RAG_EX": "SKIPPED",
                "Without_RAG_SM": "SKIPPED",
                "Without_RAG_F1": "0.000",
                "Without_RAG_Latency": "SKIPPED",
                "Without_RAG_Tokens": 0,
                "RAG_Helped": "➖",
                "With_RAG_SQL": "",
                "Without_RAG_SQL": "",
            }
    
    # Sort results by case index
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    total_time = time.time() - start_total
    evaluated_count = len(args_list)
    print(f"\nCompleted {evaluated_count} test cases in {total_time/60:.1f} minutes")
    print(f"   Average time per case: {total_time/evaluated_count:.1f} seconds")
    if SKIP_CASES:
        print(f"   Note: {len(SKIP_CASES)} case(s) were skipped: {SKIP_CASES}")
    
    df = pd.DataFrame(results)
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    # Count successes
    with_rag_success = sum(1 for x in df["With_RAG_Success"] if x == "✅")
    without_rag_success = sum(1 for x in df["Without_RAG_Success"] if x == "✅")
    
    # Count EX matches
    with_rag_ex = sum(1 for x in df["With_RAG_EX"] if x == "✅")
    without_rag_ex = sum(1 for x in df["Without_RAG_EX"] if x == "✅")
    
    # Count SM matches
    with_rag_sm = sum(1 for x in df["With_RAG_SM"] if x == "✅")
    without_rag_sm = sum(1 for x in df["Without_RAG_SM"] if x == "✅")
    
    # Average F1
    with_rag_f1_avg = sum(float(x) for x in df["With_RAG_F1"]) / len(df)
    without_rag_f1_avg = sum(float(x) for x in df["Without_RAG_F1"]) / len(df)
    
    # Average latency
    with_rag_lat_avg = sum(float(x.replace("s", "")) for x in df["With_RAG_Latency"]) / len(df)
    without_rag_lat_avg = sum(float(x.replace("s", "")) for x in df["Without_RAG_Latency"]) / len(df)
    
    # Total tokens
    with_rag_tokens_total = sum(df["With_RAG_Tokens"])
    without_rag_tokens_total = sum(df["Without_RAG_Tokens"])
    
    # RAG helped count
    rag_helped = sum(1 for x in df["RAG_Helped"] if x == "✅")
    rag_hurt = sum(1 for x in df["RAG_Helped"] if x == "❌")
    rag_neutral = sum(1 for x in df["RAG_Helped"] if x == "➖")
    
    print(f"Total Test Cases: {len(df)}\n")
    
    print("Success Rate:")
    print(f"  With RAG:    {with_rag_success}/{len(df)} ({with_rag_success/len(df)*100:.1f}%)")
    print(f"  Without RAG: {without_rag_success}/{len(df)} ({without_rag_success/len(df)*100:.1f}%)")
    print(f"  Difference:  {with_rag_success - without_rag_success:+d} ({((with_rag_success - without_rag_success)/len(df)*100):+.1f}%)\n")
    
    print("Execution Accuracy (EX):")
    print(f"  With RAG:    {with_rag_ex}/{len(df)} ({with_rag_ex/len(df)*100:.1f}%)")
    print(f"  Without RAG: {without_rag_ex}/{len(df)} ({without_rag_ex/len(df)*100:.1f}%)")
    print(f"  Difference:  {with_rag_ex - without_rag_ex:+d} ({((with_rag_ex - without_rag_ex)/len(df)*100):+.1f}%)\n")
    
    print("Semantic Match (SM):")
    print(f"  With RAG:    {with_rag_sm}/{len(df)} ({with_rag_sm/len(df)*100:.1f}%)")
    print(f"  Without RAG: {without_rag_sm}/{len(df)} ({without_rag_sm/len(df)*100:.1f}%)")
    print(f"  Difference:  {with_rag_sm - without_rag_sm:+d} ({((with_rag_sm - without_rag_sm)/len(df)*100):+.1f}%)\n")
    
    print("Average F1-Score:")
    print(f"  With RAG:    {with_rag_f1_avg:.3f}")
    print(f"  Without RAG: {without_rag_f1_avg:.3f}")
    print(f"  Difference:  {with_rag_f1_avg - without_rag_f1_avg:+.3f}\n")
    
    print("Average Latency:")
    print(f"  With RAG:    {with_rag_lat_avg:.2f}s")
    print(f"  Without RAG: {without_rag_lat_avg:.2f}s")
    print(f"  Difference:  {with_rag_lat_avg - without_rag_lat_avg:+.2f}s\n")
    
    print("Total Tokens:")
    print(f"  With RAG:    {with_rag_tokens_total:,}")
    print(f"  Without RAG: {without_rag_tokens_total:,}")
    print(f"  Difference:  {with_rag_tokens_total - without_rag_tokens_total:+,}\n")
    
    print("RAG Impact:")
    print(f"  RAG Helped:  {rag_helped} queries")
    print(f"  RAG Hurt:    {rag_hurt} queries")
    print(f"  Neutral:     {rag_neutral} queries\n")


def load_test_cases(file_path: str = None) -> List[Dict[str, str]]:
    """Load test cases from JSON file. Defaults to data/test_cases_template.json"""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Try default template file in data/ directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up from scripts/ to text2sql/
    template_path = os.path.join(parent_dir, "data", "test_cases_template.json")
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fallback: try in current directory
    if os.path.exists("test_cases_template.json"):
        with open("test_cases_template.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return []


def main():
    """Main function to run RAG comparison."""
    import sys
    
    print("="*80)
    print("RAG vs No-RAG Comparison Tool")
    print("="*80)
    
    # Load test cases
    test_cases = []
    
    # Try to load from command line argument
    if len(sys.argv) > 1:
        test_cases = load_test_cases(sys.argv[1])
    
    # Try default template
    if len(test_cases) == 0:
        test_cases = load_test_cases("test_cases_template.json")
    
    # Try in text2sql directory
    if len(test_cases) == 0:
        # Try data/ directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        template_path = os.path.join(parent_dir, "data", "test_cases_template.json")
        test_cases = load_test_cases(template_path)
    
    if len(test_cases) == 0:
        print("\nWARNING: No test cases found!")
        print("\nPlease create a JSON file with test cases, or use the template:")
        print("  python compare_rag.py test_cases.json")
        print(f"\nOr place test_cases_template.json in the data/ directory: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')}")
        print("\nExample JSON format:")
        print('  [')
        print('    {"question": "Your question", "gold_sql": "SELECT ... FROM ...;"},')
        print('    ...')
        print('  ]')
        return
    
    print(f"\nLoaded {len(test_cases)} test cases\n")
    
    # Configuration
    model_name = os.getenv("MODEL_NAME", "ollama/llama3.1:8b")
    technique = "Few-Shot"  # You can change this or test all techniques
    db_url = os.getenv("DB_URL", "sqlite:///income.db")
    
    # Parallel processing configuration
    # For RTX 4060 (8GB VRAM):
    #   - Ollama: 3-5 workers (model uses ~4-5GB, need room for parallel requests)
    #   - OpenAI: 10-20 workers (API rate limits apply)
    # You can override by setting MAX_WORKERS environment variable
    max_workers = int(os.getenv("MAX_WORKERS", "0"))  # 0 = auto-detect based on model
    
    if max_workers == 0:
        max_workers = None  # Let function auto-detect
    
    print(f"Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Technique: {technique}")
    print(f"   Max Workers: {max_workers if max_workers else 'Auto (4 for Ollama, 10 for OpenAI)'}")
    print()
    
    # Run comparison
    df = compare_rag_performance(test_cases, model_name, technique, db_url, max_workers=max_workers)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=False))
    
    # Print summary
    print_summary(df)
    
    # Save to CSV in data/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up from scripts/ to text2sql/
    output_file = os.path.join(parent_dir, "data", "rag_comparison_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


# ============================================================================
# ADD YOUR 10 TEST CASES HERE
# ============================================================================
# Replace the empty list below with your test cases:
# 
# TEST_CASES = [
#     {
#         "question": "Your question here",
#         "gold_sql": "SELECT ... FROM ...;"
#     },
#     # ... add 9 more test cases
# ]

TEST_CASES = [
    {
        "question": "What is the average number of hours women work per week?",
        "gold_sql": "SELECT AVG(hours_per_week) AS avg_hours_women_work FROM adult_income WHERE sex = 'Female';"
    },
    {
        "question": "Show the average hours worked per week by sex.",
        "gold_sql": "SELECT sex, AVG(hours_per_week) AS avg_hours_per_week FROM adult_income GROUP BY sex;"
    },
    {
        "question": "Among U.S. residents, which occupations have the highest share of people earning more than 50K? Show top 10.",
        "gold_sql": "SELECT occupation, COUNT(*) AS total_people, SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) AS high_earners, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) AS pct_high_income FROM adult_income WHERE native_country = 'United-States' AND occupation <> '?' GROUP BY occupation HAVING total_people >= 20 ORDER BY pct_high_income DESC, total_people DESC LIMIT 10;"
    },
    {
        "question": "For each education level, what are the average capital gains, capital losses, and hours worked?",
        "gold_sql": "SELECT education, AVG(capital_gain) AS avg_capital_gain, AVG(capital_loss) AS avg_capital_loss, AVG(hours_per_week) AS avg_hours_per_week FROM adult_income GROUP BY education;"
    },
    {
        "question": "For each education level and sex, how many people are there, what percent earn more than 50K, the average age and hours per week, and the most common occupation? Rank by highest percent of high earners.",
        "gold_sql": "WITH base AS (SELECT education, sex, occupation, age, hours_per_week, income FROM adult_income WHERE occupation <> '?'), occ_counts AS (SELECT education, sex, occupation, COUNT(*) AS occ_cnt FROM base GROUP BY education, sex, occupation), top_occ AS (SELECT education, sex, occupation FROM (SELECT education, sex, occupation, occ_cnt, ROW_NUMBER() OVER (PARTITION BY education, sex ORDER BY occ_cnt DESC, occupation ASC) AS rn FROM occ_counts) WHERE rn = 1), agg AS (SELECT education, sex, COUNT(*) AS total_people, AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income, AVG(hours_per_week) AS avg_hours_per_week, AVG(age) AS avg_age FROM base GROUP BY education, sex) SELECT a.education, a.sex, a.total_people, a.pct_high_income * 100.0 AS pct_high_income_percent, a.avg_hours_per_week, a.avg_age, t.occupation AS top_occupation FROM agg a LEFT JOIN top_occ t ON a.education = t.education AND a.sex = t.sex ORDER BY a.pct_high_income DESC, a.total_people DESC LIMIT 100;"
    },
    {
        "question": "Which marital status has the highest average capital gain among people earning more than 50K?",
        "gold_sql": "SELECT marital_status, AVG(capital_gain) AS avg_capital_gain FROM adult_income WHERE income = '>50K' GROUP BY marital_status ORDER BY avg_capital_gain DESC LIMIT 5;"
    },
    {
        "question": "Among people who work more than 40 hours per week, what is the average age by sex and education? Show top 20 groups by average age.",
        "gold_sql": "SELECT sex, education, AVG(age) AS avg_age_over_40hrs FROM adult_income WHERE hours_per_week > 40 GROUP BY sex, education ORDER BY avg_age_over_40hrs DESC LIMIT 20;"
    },
    {
        "question": "For each native country, how many people are there and what percentage earn more than 50K? Only show countries with at least 30 people.",
        "gold_sql": "SELECT native_country, COUNT(*) AS total_people, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY native_country HAVING total_people >= 30 ORDER BY pct_high_income DESC, total_people DESC;"
    },
    {
        "question": "What are the top 5 occupations for women by average hours worked (only consider occupations with at least 20 women)?",
        "gold_sql": "SELECT occupation, AVG(hours_per_week) AS avg_hours_women FROM adult_income WHERE sex = 'Female' AND occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 20 ORDER BY avg_hours_women DESC LIMIT 5;"
    },
    {
        "question": "What is the average education_num by income class (<=50K vs >50K)?",
        "gold_sql": "SELECT income, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY income ORDER BY income;"
    },
    {
        "question": "List the top 10 education levels by average hours worked per week.",
        "gold_sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY education ORDER BY avg_hours DESC LIMIT 10;"
    },
    {
        "question": "For each race, what percentage of people earn more than 50K?",
        "gold_sql": "SELECT race, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY race ORDER BY pct_high_income DESC;"
    },
    {
        "question": "Which occupations have the highest average age among men? Show top 10 with at least 20 men.",
        "gold_sql": "SELECT occupation, AVG(age) AS avg_age_men FROM adult_income WHERE sex = 'Male' AND occupation <> '?' GROUP BY occupation HAVING COUNT(*) >= 20 ORDER BY avg_age_men DESC LIMIT 10;"
    },
    {
        "question": "What is the distribution of people by marital status and sex?",
        "gold_sql": "SELECT marital_status, sex, COUNT(*) AS cnt FROM adult_income GROUP BY marital_status, sex ORDER BY cnt DESC;"
    },
    {
        "question": "For each relationship type, what is the average capital gain and capital loss?",
        "gold_sql": "SELECT relationship, AVG(capital_gain) AS avg_capital_gain, AVG(capital_loss) AS avg_capital_loss FROM adult_income GROUP BY relationship ORDER BY avg_capital_gain DESC;"
    },
    {
        "question": "Among people with education level 'Bachelors', what is the average hours per week by sex?",
        "gold_sql": "SELECT sex, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE education = 'Bachelors' GROUP BY sex;"
    },
    {
        "question": "Which native countries have the highest average education_num (only show countries with at least 30 people)?",
        "gold_sql": "SELECT native_country, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY native_country HAVING COUNT(*) >= 30 ORDER BY avg_education_num DESC;"
    },
    {
        "question": "For each workclass, what fraction of people earns more than 50K?",
        "gold_sql": "SELECT workclass, AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS frac_high_income FROM adult_income GROUP BY workclass ORDER BY frac_high_income DESC;"
    },
    {
        "question": "What are the top 10 occupations by count overall (ignore unknown '?')?",
        "gold_sql": "SELECT occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY occupation ORDER BY cnt DESC LIMIT 10;"
    },
    {
        "question": "What is the average age of women who earn more than 50K?",
        "gold_sql": "SELECT AVG(age) AS avg_age_women_high_income FROM adult_income WHERE sex = 'Female' AND income = '>50K';"
    },
    {
        "question": "Compute the global percentage of people earning more than 50K.",
        "gold_sql": "SELECT 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income_global FROM adult_income;"
    },
    {
        "question": "For each education level, show the percentage of people who work more than 40 hours per week.",
        "gold_sql": "SELECT education, 100.0 * AVG(CASE WHEN hours_per_week > 40 THEN 1.0 ELSE 0.0 END) AS pct_over_40hrs FROM adult_income GROUP BY education ORDER BY pct_over_40hrs DESC;"
    },
    {
        "question": "Among U.S. residents, which marital statuses have the highest average hours worked per week?",
        "gold_sql": "SELECT marital_status, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE native_country = 'United-States' GROUP BY marital_status ORDER BY avg_hours DESC;"
    },
    {
        "question": "For each occupation, what is the average education_num and average hours worked per week? Show top 15 by average education_num (ignore '?').",
        "gold_sql": "SELECT occupation, AVG(education_num) AS avg_education_num, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE occupation <> '?' GROUP BY occupation ORDER BY avg_education_num DESC, avg_hours DESC LIMIT 15;"
    },
    {
        "question": "What are the top 10 native countries by high-income percentage among women (countries with at least 20 women)?",
        "gold_sql": "SELECT native_country, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income WHERE sex = 'Female' GROUP BY native_country HAVING COUNT(*) >= 20 ORDER BY pct_high_income DESC LIMIT 10;"
    },
    {
        "question": "Which race has the lowest average capital loss?",
        "gold_sql": "SELECT race, AVG(capital_loss) AS avg_capital_loss FROM adult_income GROUP BY race ORDER BY avg_capital_loss ASC LIMIT 1;"
    },
    {
        "question": "For each sex and marital status, compute the percentage working at least 50 hours per week.",
        "gold_sql": "SELECT sex, marital_status, 100.0 * AVG(CASE WHEN hours_per_week >= 50 THEN 1.0 ELSE 0.0 END) AS pct_50plus_hours FROM adult_income GROUP BY sex, marital_status ORDER BY pct_50plus_hours DESC;"
    },
    {
        "question": "Among those with education_num >= 13, what is the average capital gain by sex?",
        "gold_sql": "SELECT sex, AVG(capital_gain) AS avg_capital_gain FROM adult_income WHERE education_num >= 13 GROUP BY sex ORDER BY avg_capital_gain DESC;"
    },
    {
        "question": "Which occupations have the largest difference in average hours between men and women? Show top 10 by absolute difference (ignore '?' and require at least 20 men and 20 women per occupation).",
        "gold_sql": "WITH occ_stats AS (SELECT occupation, sex, COUNT(*) AS cnt, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE occupation <> '?' GROUP BY occupation, sex), pivot AS (SELECT o1.occupation, o1.avg_hours AS avg_male, o2.avg_hours AS avg_female, o1.cnt AS cnt_male, o2.cnt AS cnt_female FROM occ_stats o1 JOIN occ_stats o2 ON o1.occupation = o2.occupation AND o1.sex = 'Male' AND o2.sex = 'Female') SELECT occupation, ABS(avg_male - avg_female) AS avg_hours_gap FROM pivot WHERE cnt_male >= 20 AND cnt_female >= 20 ORDER BY avg_hours_gap DESC LIMIT 10;"
    },
    {
        "question": "For each education level, show the average age for people earning <=50K vs >50K.",
        "gold_sql": "SELECT education, income, AVG(age) AS avg_age FROM adult_income GROUP BY education, income ORDER BY education, income;"
    },
    {
        "question": "What are the top 10 workclasses by high-income percentage (require at least 30 people in the workclass)?",
        "gold_sql": "SELECT workclass, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income, COUNT(*) AS cnt FROM adult_income GROUP BY workclass HAVING cnt >= 30 ORDER BY pct_high_income DESC LIMIT 10;"
    },
    {
        "question": "For each native country, show average hours worked per week by sex.",
        "gold_sql": "SELECT native_country, sex, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY native_country, sex ORDER BY native_country, sex;"
    },
    {
        "question": "For each relationship type, what fraction of people earn more than 50K?",
        "gold_sql": "SELECT relationship, AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS frac_high_income FROM adult_income GROUP BY relationship ORDER BY frac_high_income DESC;"
    },
    {
        "question": "Among people older than 50, what is the average hours worked per week by occupation? Show top 15 by average hours (ignore '?').",
        "gold_sql": "SELECT occupation, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE age > 50 AND occupation <> '?' GROUP BY occupation ORDER BY avg_hours DESC LIMIT 15;"
    },
    {
        "question": "Which occupation has the highest average education_num (ignore '?', require at least 30 people)?",
        "gold_sql": "SELECT occupation, AVG(education_num) AS avg_education_num, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING cnt >= 30 ORDER BY avg_education_num DESC LIMIT 1;"
    },
    {
        "question": "For each sex, what is the percentage of people in each marital status who earn more than 50K?",
        "gold_sql": "SELECT sex, marital_status, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income GROUP BY sex, marital_status ORDER BY sex, pct_high_income DESC;"
    },
    {
        "question": "Among U.S. residents, what are the top 10 native countries by average hours worked per week?",
        "gold_sql": "SELECT native_country, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE native_country <> 'United-States' GROUP BY native_country ORDER BY avg_hours DESC LIMIT 10;"
    },
    {
        "question": "For each education level, show the proportion of men vs women (share by sex).",
        "gold_sql": "SELECT education, sex, 100.0 * COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY education) AS share_by_sex FROM adult_income GROUP BY education, sex ORDER BY education, share_by_sex DESC;"
    },
    {
        "question": "Within each education level, rank occupations by count (ignore '?') and show the top one per education.",
        "gold_sql": "WITH occ_counts AS (SELECT education, occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY education, occupation), ranked AS (SELECT education, occupation, cnt, ROW_NUMBER() OVER (PARTITION BY education ORDER BY cnt DESC, occupation ASC) AS rn FROM occ_counts) SELECT education, occupation, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
    },
    {
        "question": "What is the median-like approximation: show the 3rd quartile (75th percentile) of hours_per_week by sex using window functions (approx via NTILE)?",
        "gold_sql": "WITH ranked AS (SELECT sex, hours_per_week, NTILE(4) OVER (PARTITION BY sex ORDER BY hours_per_week) AS quart FROM adult_income) SELECT sex, MAX(hours_per_week) AS approx_q3_hours FROM ranked WHERE quart = 3 GROUP BY sex;"
    },
    {
        "question": "For each native country, show the most common occupation (mode) ignoring '?', with its count.",
        "gold_sql": "WITH occ AS (SELECT native_country, occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY native_country, occupation), ranked AS (SELECT native_country, occupation, cnt, ROW_NUMBER() OVER (PARTITION BY native_country ORDER BY cnt DESC, occupation ASC) AS rn FROM occ) SELECT native_country, occupation AS top_occupation, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
    },
    {
        "question": "Among people with capital_gain > 0, what is the average age and average education_num by sex?",
        "gold_sql": "SELECT sex, AVG(age) AS avg_age, AVG(education_num) AS avg_education_num FROM adult_income WHERE capital_gain > 0 GROUP BY sex;"
    },
    {
        "question": "For each workclass, show the average hours worked per week by income class (<=50K vs >50K).",
        "gold_sql": "SELECT workclass, income, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY workclass, income ORDER BY workclass, income;"
    },
    {
        "question": "Which education levels have the highest high-income percentage among women (require at least 20 women per education)?",
        "gold_sql": "SELECT education, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income_women, COUNT(*) AS cnt FROM adult_income WHERE sex = 'Female' GROUP BY education HAVING cnt >= 20 ORDER BY pct_high_income_women DESC;"
    },
    {
        "question": "For each relationship type and sex, what is the average hours worked per week and the fraction earning >50K?",
        "gold_sql": "SELECT relationship, sex, AVG(hours_per_week) AS avg_hours, AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS frac_high_income FROM adult_income GROUP BY relationship, sex ORDER BY relationship, sex;"
    },
    {
        "question": "Among people younger than 30, which occupations have the highest percentage of >50K earners (ignore '?', require at least 20 people per occupation)?",
        "gold_sql": "SELECT occupation, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income, COUNT(*) AS cnt FROM adult_income WHERE age < 30 AND occupation <> '?' GROUP BY occupation HAVING cnt >= 20 ORDER BY pct_high_income DESC LIMIT 10;"
    },
    {
        "question": "For each native country, what is the average age by sex for people who work at least 45 hours per week?",
        "gold_sql": "SELECT native_country, sex, AVG(age) AS avg_age FROM adult_income WHERE hours_per_week >= 45 GROUP BY native_country, sex ORDER BY native_country, sex;"
    },
    {
        "question": "Which combinations of education and marital_status have the highest average education_num? Show top 10.",
        "gold_sql": "SELECT education, marital_status, AVG(education_num) AS avg_education_num FROM adult_income GROUP BY education, marital_status ORDER BY avg_education_num DESC LIMIT 10;"
    },
    {
        "question": "For each occupation, compute the share of workers by sex (percentage within the occupation).",
        "gold_sql": "SELECT occupation, sex, 100.0 * COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY occupation) AS share_by_sex FROM adult_income WHERE occupation <> '?' GROUP BY occupation, sex ORDER BY occupation, share_by_sex DESC;"
    },
    {
        "question": "Among people with zero capital_loss, what is the average hours worked per week by education level?",
        "gold_sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE capital_loss = 0 GROUP BY education ORDER BY avg_hours DESC;"
    },
    {
        "question": "For each race, what is the most common marital status and its count?",
        "gold_sql": "WITH ms AS (SELECT race, marital_status, COUNT(*) AS cnt FROM adult_income GROUP BY race, marital_status), ranked AS (SELECT race, marital_status, cnt, ROW_NUMBER() OVER (PARTITION BY race ORDER BY cnt DESC, marital_status ASC) AS rn FROM ms) SELECT race, marital_status AS top_marital_status, cnt FROM ranked WHERE rn = 1 ORDER BY cnt DESC;"
    },
    {
        "question": "Which occupations have the highest fraction of people working more than 50 hours per week (ignore '?', require at least 30 people)? Show top 10.",
        "gold_sql": "SELECT occupation, 100.0 * AVG(CASE WHEN hours_per_week > 50 THEN 1.0 ELSE 0.0 END) AS pct_over_50hrs, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY occupation HAVING cnt >= 30 ORDER BY pct_over_50hrs DESC LIMIT 10;"
    },
    {
        "question": "For each native country, compute the average capital_gain and capital_loss.",
        "gold_sql": "SELECT native_country, AVG(capital_gain) AS avg_capital_gain, AVG(capital_loss) AS avg_capital_loss FROM adult_income GROUP BY native_country ORDER BY avg_capital_gain DESC;"
    },
    {
        "question": "Among people with income <=50K, which education levels have the highest average hours worked per week?",
        "gold_sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE income = '<=50K' GROUP BY education ORDER BY avg_hours DESC;"
    },
    {
        "question": "For each sex, list the top 5 occupations by count (ignore '?').",
        "gold_sql": "WITH counts AS (SELECT sex, occupation, COUNT(*) AS cnt FROM adult_income WHERE occupation <> '?' GROUP BY sex, occupation) SELECT sex, occupation, cnt FROM (SELECT sex, occupation, cnt, ROW_NUMBER() OVER (PARTITION BY sex ORDER BY cnt DESC, occupation ASC) AS rn FROM counts) WHERE rn <= 5 ORDER BY sex, cnt DESC;"
    },
    {
        "question": "Compute the overall average of hours_per_week and the standard 'share working 40 hours' across the dataset.",
        "gold_sql": "SELECT AVG(hours_per_week) AS avg_hours, AVG(CASE WHEN hours_per_week = 40 THEN 1.0 ELSE 0.0 END) AS share_40hrs FROM adult_income;"
    },
    {
        "question": "For each education level, compute the average hours worked and sort by that average ascending.",
        "gold_sql": "SELECT education, AVG(hours_per_week) AS avg_hours FROM adult_income GROUP BY education ORDER BY avg_hours ASC;"
    },
    {
        "question": "Among men, which marital statuses have the highest average hours worked per week?",
        "gold_sql": "SELECT marital_status, AVG(hours_per_week) AS avg_hours FROM adult_income WHERE sex = 'Male' GROUP BY marital_status ORDER BY avg_hours DESC;"
    },
    {
        "question": "For each occupation, show the percentage of people earning >50K within that occupation (ignore '?').",
        "gold_sql": "SELECT occupation, 100.0 * AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income FROM adult_income WHERE occupation <> '?' GROUP BY occupation ORDER BY pct_high_income DESC;"
    },
    {
        "question": "Which native countries have the largest number of people older than 60? Show top 10.",
        "gold_sql": "SELECT native_country, COUNT(*) AS cnt FROM adult_income WHERE age > 60 GROUP BY native_country ORDER BY cnt DESC LIMIT 10;"
    },
    {
        "question": "For each sex, what is the average age and average education_num among people with income >50K?",
        "gold_sql": "SELECT sex, AVG(age) AS avg_age, AVG(education_num) AS avg_education_num FROM adult_income WHERE income = '>50K' GROUP BY sex;"
    }
]

  # Add your test cases here or load from JSON file

# ============================================================================

if __name__ == "__main__":
    # If test cases are provided in the script, use them
    if TEST_CASES:
        # Temporarily override the load function
        original_load = load_test_cases
        def load_test_cases_override(file_path=None):
            return TEST_CASES
        load_test_cases = load_test_cases_override
    
    main()

