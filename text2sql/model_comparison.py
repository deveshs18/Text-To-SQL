"""Model comparison utilities: GPT vs Ollama side-by-side evaluation."""
import os
import time
from typing import Dict, List, Optional, Tuple
from db import get_engine, execute_sql
from schema_retriever import get_schema_snippet
from prompts import build_few_shot_prompt, build_cot_prompt, build_ltm_prompt, build_eg_prompt
from model_client import generate_sql, refine_sql
from advanced_metrics import calculate_all_metrics

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    memory = {
        "cpu_memory_mb": 0.0,
        "cpu_memory_percent": 0.0,
        "gpu_memory_mb": 0.0,
        "gpu_memory_percent": 0.0
    }
    
    if psutil:
        memory["cpu_memory_mb"] = psutil.virtual_memory().used / (1024 * 1024)
        memory["cpu_memory_percent"] = psutil.virtual_memory().percent
    
    # GPU memory if available
    if torch and torch.cuda.is_available():
        memory["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory["gpu_memory_percent"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
    
    return memory


def calculate_cost(tokens: int, model_name: str) -> float:
    """
    Calculate cost in USD based on token usage and model pricing.
    Returns cost in USD.
    """
    if model_name.startswith("openai/"):
        model = model_name.replace("openai/", "")
        
        # OpenAI pricing per 1M tokens (as of 2024)
        pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
        
        # Default to gpt-4o-mini pricing if model not found
        price = pricing.get(model, pricing["gpt-4o-mini"])
        
        # Assume 70% input, 30% output tokens
        input_cost = (tokens * 0.7 / 1_000_000) * price["input"]
        output_cost = (tokens * 0.3 / 1_000_000) * price["output"]
        
        return input_cost + output_cost
    
    elif model_name.startswith("ollama/"):
        # Ollama is free (local)
        return 0.0
    
    return 0.0


def evaluate_single_query(
    question: str,
    gold_sql: str,
    model_name: str,
    technique: str,
    db_url: str,
    table_name: str = "adult_income"
) -> Dict:
    """
    Evaluate a single query with one model and technique.
    Returns comprehensive metrics dictionary.
    """
    engine = get_engine(db_url)
    schema = get_schema_snippet(question, engine, table_name)
    
    # Build prompt based on technique
    prompt_builders = {
        "Few-Shot": build_few_shot_prompt,
        "CoT": build_cot_prompt,
        "LtM": build_ltm_prompt,
        "EG": build_eg_prompt
    }
    
    prompt = prompt_builders[technique](schema, question)
    
    # Get memory before
    memory_before = get_memory_usage()
    
    # Generate SQL
    start_time = time.time()
    try:
        sql, generation_metrics = generate_sql(
            prompt,
            model_name,
            temperature=0.1
        )
        
        # Get memory after
        memory_after = get_memory_usage()
        memory_delta = {
            "cpu_memory_mb": memory_after["cpu_memory_mb"] - memory_before["cpu_memory_mb"],
            "gpu_memory_mb": memory_after["gpu_memory_mb"] - memory_before["gpu_memory_mb"]
        }
        
        # Validate SQL
        success = False
        error = None
        df = None
        exec_latency = 0.0
        
        try:
            exec_success, df, exec_error, exec_latency = execute_sql(sql, engine)
            success = exec_success
            error = exec_error
        except Exception as e:
            error = str(e)
        
        # Calculate all metrics
        all_metrics = calculate_all_metrics(
            gold_sql,
            sql,
            engine,
            include_execution=True
        )
        
        # Calculate cost
        cost = calculate_cost(generation_metrics["tokens_used"], model_name)
        
        result = {
            "success": success,
            "sql": sql,
            "error": error,
            "latency": generation_metrics["latency"],
            "execution_latency": exec_latency,
            "total_latency": generation_metrics["latency"] + exec_latency,
            "tokens_used": generation_metrics["tokens_used"],
            "input_tokens": generation_metrics["input_tokens"],
            "output_tokens": generation_metrics["output_tokens"],
            "cost_usd": cost,
            "memory_delta_mb": memory_delta,
            "metrics": all_metrics
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "sql": "",
            "error": str(e),
            "latency": time.time() - start_time,
            "execution_latency": 0.0,
            "total_latency": time.time() - start_time,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "memory_delta_mb": {"cpu_memory_mb": 0.0, "gpu_memory_mb": 0.0},
            "metrics": {}
        }


def compare_models(
    question: str,
    gold_sql: str,
    models: List[str],
    techniques: List[str],
    db_url: str,
    table_name: str = "adult_income"
) -> Dict:
    """
    Compare multiple models on the same query.
    Returns comparison results dictionary.
    """
    results = {}
    
    for model in models:
        results[model] = {}
        for technique in techniques:
            result = evaluate_single_query(
                question,
                gold_sql,
                model,
                technique,
                db_url,
                table_name
            )
            results[model][technique] = result
    
    return results


def batch_evaluate(
    test_cases: List[Dict[str, str]],
    models: List[str],
    techniques: List[str],
    db_url: str,
    table_name: str = "adult_income"
) -> Dict:
    """
    Batch evaluate multiple test cases.
    test_cases: List of {"question": "...", "gold_sql": "..."}
    Returns aggregated results.
    """
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        gold_sql = test_case["gold_sql"]
        
        print(f"Evaluating test case {i+1}/{len(test_cases)}: {question[:50]}...")
        
        case_results = compare_models(
            question,
            gold_sql,
            models,
            techniques,
            db_url,
            table_name
        )
        
        all_results.append({
            "question": question,
            "gold_sql": gold_sql,
            "results": case_results
        })
    
    # Aggregate metrics
    aggregated = aggregate_results(all_results)
    
    return {
        "test_cases": all_results,
        "aggregated": aggregated
    }


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Aggregate metrics across all test cases."""
    aggregated = {}
    
    for result in all_results:
        for model, techniques in result["results"].items():
            if model not in aggregated:
                aggregated[model] = {}
            
            for technique, metrics in techniques.items():
                if technique not in aggregated[model]:
                    aggregated[model][technique] = {
                        "em": [],
                        "ex": [],
                        "f1": [],
                        "bleu": [],
                        "rouge_l": [],
                        "latency": [],
                        "cost": [],
                        "success_rate": []
                    }
                
                agg = aggregated[model][technique]
                
                if metrics.get("success", False):
                    agg["success_rate"].append(1)
                else:
                    agg["success_rate"].append(0)
                
                if "metrics" in metrics and metrics["metrics"]:
                    m = metrics["metrics"]
                    agg["em"].append(1 if m.get("em", False) else 0)
                    agg["ex"].append(1 if m.get("execution_accuracy", False) else 0)
                    agg["f1"].append(m.get("f1_score", 0.0))
                    agg["bleu"].append(m.get("bleu_score", 0.0))
                    agg["rouge_l"].append(m.get("rouge_l_score", 0.0))
                
                agg["latency"].append(metrics.get("total_latency", 0.0))
                agg["cost"].append(metrics.get("cost_usd", 0.0))
    
    # Calculate averages
    for model in aggregated:
        for technique in aggregated[model]:
            agg = aggregated[model][technique]
            
            # Convert to averages
            def avg(lst):
                return sum(lst) / len(lst) if lst else 0.0
            
            aggregated[model][technique] = {
                "em_avg": avg(agg["em"]),
                "ex_avg": avg(agg["ex"]),
                "f1_avg": avg(agg["f1"]),
                "bleu_avg": avg(agg["bleu"]),
                "rouge_l_avg": avg(agg["rouge_l"]),
                "latency_avg": avg(agg["latency"]),
                "cost_total": sum(agg["cost"]),
                "success_rate": avg(agg["success_rate"]),
                "num_tests": len(agg["em"])
            }
    
    return aggregated

