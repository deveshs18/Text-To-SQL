"""Streamlit UI for Text-to-SQL with multiple prompting techniques."""
import asyncio
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from db import get_engine, validate_sql, execute_sql
from schema_retriever import get_schema_snippet, expand_schema_snippet
from model_client import generate_sql, refine_sql, generate_sql_async, refine_sql_async
from text2sql.scripts.prompts import (
    build_few_shot_prompt,
    build_cot_prompt,
    build_ltm_prompt,
    build_eg_prompt,
    build_refine_prompt
)
from metrics import exact_match, execution_accuracy
from advanced_metrics import calculate_all_metrics

load_dotenv()

# Page config
st.set_page_config(
    page_title="Text-to-SQL",
    page_icon="🗄️",
    layout="wide"
)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "saved_results" not in st.session_state:
    st.session_state.saved_results = []


def compute_score(success: bool, df: pd.DataFrame, latency: float, execution_match: bool = None) -> float:
    """
    Compute technique score prioritizing correctness.
    - If execution_match is provided (evaluation mode): 200 if correct, 0 if wrong
    - Otherwise: 100 if success, 0 if failed
    - Bonus: +10 for non-empty results
    - Penalty: -5 for high latency
    """
    # If we have evaluation data, prioritize correctness
    if execution_match is not None:
        if not execution_match:
            return 0.0  # Wrong answer = 0 points
        score = 200.0  # Correct answer = high score
    else:
        # No evaluation data, just check if it ran
        if not success:
            return 0.0
        score = 100.0
    
    # Bonus for non-empty results
    if df is not None and not df.empty:
        score += 10.0
    
    # Small latency penalty (max -5 points for >5 seconds)
    if latency > 5.0:
        score -= 5.0
    elif latency > 2.0:
        score -= 2.0 * (latency - 2.0) / 3.0
    
    return max(0.0, score)


def find_consensus(results: dict) -> list:
    """Find techniques that returned identical results."""
    successful = {
        name: res for name, res in results.items()
        if res["success"] and res["df"] is not None and not res["df"].empty
    }
    
    if len(successful) < 2:
        return []
    
    consensus = []
    names = list(successful.keys())
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            df1 = successful[name1]["df"]
            df2 = successful[name2]["df"]
            
            # Compare dataframes (simplified: string representation)
            try:
                if df1.to_csv() == df2.to_csv():
                    if name1 not in consensus:
                        consensus.append(name1)
                    if name2 not in consensus:
                        consensus.append(name2)
            except:
                pass
    
    return list(set(consensus))


def saved_results_tab():
    """Tab for viewing and comparing saved query results."""
    st.header("📊 Saved Results & Model Comparison")
    
    if not st.session_state.saved_results:
        st.info("No saved results yet. Enable 'Evaluation Mode' and run queries in the 'Query & Generate' tab to save results here.")
        return
    
    # Summary stats
    total_queries = len(st.session_state.saved_results)
    unique_models = list(set(r["model_name"] for r in st.session_state.saved_results))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Saved Queries", total_queries)
    with col2:
        st.metric("Unique Models", len(unique_models))
    with col3:
        st.metric("Models Used", ", ".join(unique_models) if len(unique_models) <= 2 else f"{len(unique_models)} models")
    
    st.markdown("---")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_model = st.selectbox(
            "Filter by Model",
            options=["All"] + unique_models,
            index=0
        )
    
    with filter_col2:
        selected_technique = st.selectbox(
            "Filter by Technique",
            options=["All", "Few-Shot", "CoT", "LtM", "EG"],
            index=0
        )
    
    # Prepare data for comparison table
    comparison_data = []
    for entry in st.session_state.saved_results:
        if selected_model != "All" and entry["model_name"] != selected_model:
            continue
            
        for technique, tech_data in entry["techniques"].items():
            if selected_technique != "All" and technique != selected_technique:
                continue
                
            comparison_data.append({
                "Question": entry["question"][:50] + "..." if len(entry["question"]) > 50 else entry["question"],
                "Model": entry["model_name"],
                "Technique": technique,
                "Timestamp": entry["timestamp"],
                "EX": "✅" if tech_data.get("execution_accuracy") else "❌",
                "SM": "✅" if tech_data.get("semantic_match") else "❌",
                "F1": f"{tech_data.get('f1_score', 0.0):.3f}",
                "BLEU": f"{tech_data.get('bleu_score', 0.0):.3f}",
                "ROUGE-L": f"{tech_data.get('rouge_l_score', 0.0):.3f}",
                "Latency (s)": f"{tech_data.get('latency', 0.0):.2f}",
                "Tokens": tech_data.get("tokens_used", 0),
                "Cost (USD)": f"${tech_data.get('cost_usd', 0.0):.6f}" if tech_data.get('cost_usd', 0.0) > 0 else "Free",
                "Success": "✅" if tech_data.get("success") else "❌"
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Model comparison summary
        if len(unique_models) >= 2:
            st.markdown("---")
            st.subheader("📈 Model Comparison Summary")
            
            summary_data = []
            for model in unique_models:
                for tech in ["Few-Shot", "CoT", "LtM", "EG"]:
                    model_tech_data = [d for d in comparison_data if d["Model"] == model and d["Technique"] == tech]
                    if not model_tech_data:
                        continue
                    
                    total = len(model_tech_data)
                    ex_count = sum(1 for d in model_tech_data if d["EX"] == "✅")
                    sm_count = sum(1 for d in model_tech_data if d["SM"] == "✅")
                    avg_f1 = sum(float(d["F1"]) for d in model_tech_data) / total
                    
                    costs = [float(d["Cost (USD)"].replace("$", "").replace("Free", "0")) for d in model_tech_data]
                    avg_cost = sum(costs) / total if costs else 0.0
                    
                    latencies = [float(d["Latency (s)"]) for d in model_tech_data]
                    avg_latency = sum(latencies) / total if latencies else 0.0
                    
                    summary_data.append({
                        "Model": model,
                        "Technique": tech,
                        "EX Match %": f"{(ex_count/total*100):.1f}%",
                        "SM Match %": f"{(sm_count/total*100):.1f}%",
                        "Avg F1": f"{avg_f1:.3f}",
                        "Avg Cost": f"${avg_cost:.6f}" if avg_cost > 0 else "Free",
                        "Avg Latency": f"{avg_latency:.2f}s"
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Clear saved results button
        st.markdown("---")
        if st.button("🗑️ Clear All Saved Results", type="secondary"):
            st.session_state.saved_results = []
            st.rerun()
    else:
        st.info("No results match the selected filters.")


def main():
    st.title("🗄️ Text-to-SQL: LtM • EG • Few-Shot • CoT")
    st.markdown("Compare four prompting techniques for natural language to SQL conversion.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Query & Generate", "📊 Saved Results & Comparison"])
    
    with tab1:
        main_query_tab()
    
    with tab2:
        saved_results_tab()


async def process_technique(technique, prompt, model_name, engine, eg_auto_refine, schema, question):
    """Process a single technique asynchronously."""
    sql = ""
    refine_attempts = 0
    try:
        # Generate SQL
        temperature = 0.2 if technique == "CoT" else 0.1
        generation_metrics = {}
        try:
            sql, generation_metrics = await generate_sql_async(prompt, model_name, temperature=temperature)
        except ValueError as ve:
            # SQL extraction failed (doesn't start with SELECT/WITH)
            # For EG, trigger refine; for others, treat as error
            if technique == "EG" and eg_auto_refine:
                # Try to refine with error message
                try:
                    refine_prompt = build_refine_prompt(
                        schema, question, "", f"Model output invalid: {str(ve)}"
                    )
                    sql, refine_metrics = await refine_sql_async(refine_prompt, model_name, temperature=0.2)
                    generation_metrics.update(refine_metrics)
                    refine_attempts += 1
                except Exception:
                    return technique, {
                        "success": False,
                        "sql": "",
                        "df": None,
                        "error": f"SQL extraction failed: {str(ve)}",
                        "latency": 0.0,
                        "score": 0.0,
                        "refine_attempts": refine_attempts,
                        "tokens_used": generation_metrics.get("tokens_used", 0),
                        "cost_usd": 0.0
                    }
            else:
                return technique, {
                    "success": False,
                    "sql": "",
                    "df": None,
                    "error": f"SQL extraction failed: {str(ve)}",
                    "latency": 0.0,
                    "score": 0.0,
                    "refine_attempts": 0,
                    "tokens_used": 0,
                    "cost_usd": 0.0
                }
        
        # Validate
        is_valid, error = validate_sql(sql)
        
        if not is_valid:
            return technique, {
                "success": False,
                "sql": sql,
                "df": None,
                "error": error,
                "latency": 0.0,
                "score": 0.0,
                "refine_attempts": refine_attempts,
                "tokens_used": generation_metrics.get("tokens_used", 0),
                "cost_usd": 0.0
            }
        
        # Execute
        success, df, err, exec_latency = execute_sql(sql, engine)
        total_latency = generation_metrics.get("latency", 0.0) + exec_latency
        
        # Calculate cost (simplified)
        tokens_used = generation_metrics.get("tokens_used", 0)
        cost_usd = 0.0
        if model_name.startswith("openai/"):
            if "gpt-4o-mini" in model_name:
                cost_usd = (tokens_used * 0.7 / 1_000_000) * 0.15 + (tokens_used * 0.3 / 1_000_000) * 0.60
        
        # EG auto-refine if enabled
        if technique == "EG" and eg_auto_refine and (not success or df is None or df.empty):
            # Try refinement (max 2 attempts total)
            for attempt in range(2 - refine_attempts):
                try:
                    refine_prompt = build_refine_prompt(
                        schema, question, sql, err or "Empty result"
                    )
                    refined_sql, refine_metrics = await refine_sql_async(refine_prompt, model_name, temperature=0.2)
                    generation_metrics["tokens_used"] += refine_metrics.get("tokens_used", 0)
                    generation_metrics["latency"] += refine_metrics.get("latency", 0.0)
                    
                    is_valid_refined, error_refined = validate_sql(refined_sql)
                    if is_valid_refined:
                        success_refined, df_refined, err_refined, latency_refined = execute_sql(refined_sql, engine)
                        refine_attempts += 1
                        sql = refined_sql
                        success = success_refined
                        df = df_refined
                        err = err_refined
                        exec_latency = latency_refined
                        total_latency = generation_metrics.get("latency", 0.0) + exec_latency
                        
                        if success and df is not None and not df.empty:
                            break
                except ValueError as refine_ve:
                    err = f"Refinement failed: {str(refine_ve)}"
                    break
                except Exception as refine_err:
                    err = f"Refinement error: {str(refine_err)}"
                    break
        
        score = compute_score(success, df, total_latency)
        
        return technique, {
            "success": success,
            "sql": sql,
            "df": df,
            "error": err,
            "latency": total_latency,
            "generation_latency": generation_metrics.get("latency", 0.0),
            "execution_latency": exec_latency,
            "score": score,
            "refine_attempts": refine_attempts,
            "tokens_used": tokens_used,
            "input_tokens": generation_metrics.get("input_tokens", 0),
            "output_tokens": generation_metrics.get("output_tokens", 0),
            "cost_usd": cost_usd
        }
        
    except Exception as e:
        return technique, {
            "success": False,
            "sql": sql if sql else "",
            "df": None,
            "error": str(e),
            "latency": 0.0,
            "score": 0.0,
            "refine_attempts": refine_attempts,
            "tokens_used": 0,
            "cost_usd": 0.0
        }


def main_query_tab():
    """Main tab for querying and generating SQL."""
    # Settings sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.text_input(
            "Model Name",
            value=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
            help="Format: 'ollama/arctic-finetuned' (port 11437) or 'openai/gpt-4o-mini'"
        )
        db_url = st.text_input(
            "Database URL",
            value=os.getenv("DB_URL", "sqlite:///income.db"),
            help="SQLite path or connection string"
        )
        eg_auto_refine = st.checkbox("EG Auto-Refine", value=True, help="Automatically refine failed EG queries")
        disable_rag = st.checkbox("🚫 Disable RAG (No Schema)", value=False, help="Hide schema from model to test performance without RAG")
        
        st.markdown("---")
        st.markdown("### 📊 Evaluation Mode")
        eval_mode = st.checkbox("Enable Evaluation", value=False)
        gold_sql = ""
        if eval_mode:
            gold_sql = st.text_area(
                "Gold SQL",
                height=100,
                help="Paste the correct SQL query for comparison"
            )
    
    # Main input
    question = st.text_input(
        "💬 Natural Language Question",
        placeholder="e.g., Average hours_per_week by education (top 10)",
        key="question_input"
    )
    
    if st.button("🚀 Generate SQL", type="primary"):
        if not question:
            st.error("Please enter a question")
            return
        
        # Get schema
        try:
            engine = get_engine(db_url)
            
            if disable_rag:
                # Even when RAG is disabled, provide table name so model knows which table to query
                # But don't provide column names - this tests model's ability to infer structure
                from schema_retriever import get_all_tables
                tables = get_all_tables(engine)
                if tables:
                    # Use first table (or could be smarter about table selection)
                    table_name = tables[0]
                    schema = f"{table_name}(...)"  # Minimal schema: just table name, no columns
                else:
                    schema = ""
                st.info("🚫 RAG Disabled: Only table name provided. Model must infer column names from the question.")
            else:
                schema = get_schema_snippet(question, engine)
                
                with st.expander("📋 Schema Snippet", expanded=False):
                    st.code(schema)
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return
        
        # Build prompts
        prompts = {
            "Few-Shot": build_few_shot_prompt(schema, question, model_name=model_name),
            "CoT": build_cot_prompt(schema, question, model_name=model_name),
            "LtM": build_ltm_prompt(schema, question, model_name=model_name),
            "EG": build_eg_prompt(schema, question, model_name=model_name)
        }
        
        # Generate SQL for each technique
        results = {}
        
        with st.spinner("Generating SQL with all techniques..."):
            async def run_parallel_generation():
                tasks = []
                for technique, prompt in prompts.items():
                    tasks.append(process_technique(
                        technique, prompt, model_name, engine, eg_auto_refine, schema, question
                    ))
                results_list = await asyncio.gather(*tasks)
                return dict(results_list)
            
            # Run async loop
            try:
                results = asyncio.run(run_parallel_generation())
            except Exception as e:
                st.error(f"Async execution failed: {e}")
                results = {}
        
        # Find best technique
        best_technique = max(results.keys(), key=lambda k: results[k]["score"])
        best_score = results[best_technique]["score"]
        
        # Check consensus
        consensus = find_consensus(results)
        
        # Execute and store Gold SQL results (if in evaluation mode) - BEFORE technique results
        gold_df = None
        gold_success = False
        gold_latency = 0.0
        if eval_mode and gold_sql and gold_sql.strip():
            st.markdown("---")
            st.subheader("📊 Gold SQL (Expected Results)")
            st.code(gold_sql, language="sql")
            
            # Execute gold SQL and show results
            try:
                gold_success, gold_df, gold_err, gold_latency = execute_sql(gold_sql, engine, add_limit=False)
                if gold_success:
                    if gold_df is not None and not gold_df.empty:
                        st.markdown("**Gold SQL Results:**")
                        st.dataframe(gold_df, use_container_width=True)
                        st.caption(f"Rows: {len(gold_df)} | Execution time: {gold_latency:.3f}s")
                    elif gold_df is not None and gold_df.empty:
                        st.info("ℹ️ Gold SQL executed successfully but returned no results (empty dataframe).")
                    else:
                        st.warning("⚠️ Gold SQL executed but returned None. Check the query.")
                else:
                    st.error(f"❌ Gold SQL execution error: {gold_err}")
                    st.code(gold_sql, language="sql")
            except Exception as e:
                st.error(f"❌ Error executing gold SQL: {str(e)}")
                st.code(gold_sql, language="sql")
        
        # Display results
        st.markdown("---")
        
        if best_score > 0:
            st.success(f"🏆 **Best Technique: {best_technique}** (Score: {best_score:.1f})")
        else:
            st.warning("⚠️ All techniques failed. Check errors below or try refining your question.")
        
        if consensus:
            st.info(f"✅ Higher confidence: {', '.join(consensus)} techniques returned identical results (consensus)")
        
        # Display each technique
        cols = st.columns(2)
        
        for idx, (technique, res) in enumerate(results.items()):
            col = cols[idx % 2]
            
            with col:
                st.subheader(f"{technique}")
                
                # Score and latency
                score_emoji = "✅" if res["success"] else "❌"
                tokens_display = f" | Tokens: {res.get('tokens_used', 0):,}" if res.get('tokens_used', 0) > 0 else ""
                cost_display = f" | Cost: ${res.get('cost_usd', 0.0):.6f}" if res.get('cost_usd', 0.0) > 0 else ""
                st.caption(f"{score_emoji} Score: {res['score']:.1f} | Latency: {res['latency']:.2f}s{tokens_display}{cost_display}")
                
                if res["refine_attempts"] > 0:
                    st.caption(f"🔄 Refined {res['refine_attempts']} time(s)")
                
                # SQL
                st.code(res["sql"], language="sql")
                
                # Results or error
                if res["success"]:
                    if res["df"] is not None and not res["df"].empty:
                        # If evaluation mode and gold SQL exists, show comparison
                        if eval_mode and gold_sql and gold_success and gold_df is not None:
                            # Check if results match (with floating point tolerance, ignore column name differences)
                            from metrics import compare_dataframes
                            results_match = compare_dataframes(gold_df, res["df"], rtol=1e-5, atol=1e-8, ignore_column_names=True)
                            
                            # Show comparison
                            if results_match:
                                st.success("✅ Results Match Gold SQL!")
                            else:
                                st.warning("⚠️ Results differ from Gold SQL")
                            
                            # Side-by-side comparison
                            st.markdown("**Results Comparison:**")
                            comp_cols = st.columns(2)
                            
                            with comp_cols[0]:
                                st.markdown("**🔶 Gold SQL Results:**")
                                st.dataframe(gold_df, use_container_width=True)
                                st.caption(f"Rows: {len(gold_df)}")
                            
                            with comp_cols[1]:
                                st.markdown(f"**🔷 {technique} Results:**")
                                st.dataframe(res["df"], use_container_width=True)
                                st.caption(f"Rows: {len(res['df'])}")
                            
                            # Show differences summary
                            if not results_match:
                                try:
                                    # Try to highlight differences
                                    if len(gold_df.columns) == len(res["df"].columns) and list(gold_df.columns) == list(res["df"].columns):
                                        # Same columns - compare row counts
                                        if len(gold_df) != len(res["df"]):
                                            st.info(f"📊 Row count difference: Gold has {len(gold_df)} rows, {technique} has {len(res['df'])} rows")
                                        else:
                                            st.info("📊 Same row count but data differs. Check column values.")
                                except:
                                    pass
                        else:
                            # Normal display when not in eval mode
                            st.dataframe(res["df"], use_container_width=True)
                    else:
                        st.info("Query executed successfully but returned no results.")
                else:
                    st.error(f"Error: {res['error']}")
                
                # Comprehensive evaluation metrics (if enabled)
                if eval_mode and gold_sql and res["sql"]:
                    try:
                        all_metrics = calculate_all_metrics(
                            gold_sql,
                            res["sql"],
                            engine,
                            include_execution=True
                        )
                        
                        # Display all metrics
                        metrics_cols = st.columns(4)
                        
                        with metrics_cols[0]:
                            if all_metrics.get("em", False):
                                st.success("✅ EM: Match")
                            else:
                                st.info("❌ EM: No Match")
                        
                        with metrics_cols[1]:
                            if all_metrics.get("execution_accuracy", False):
                                st.success("✅ EX: Match")
                            else:
                                st.info("❌ EX: No Match")
                        
                        with metrics_cols[2]:
                            sm = all_metrics.get("semantic_match", False)
                            if sm:
                                st.success("✅ SM: Match")
                            else:
                                st.info("❌ SM: No Match")
                        
                        with metrics_cols[3]:
                            f1 = all_metrics.get("f1_score", 0.0)
                            st.metric("F1-Score", f"{f1:.3f}")
                        
                        # Additional metrics in expander
                        with st.expander("📊 Detailed Metrics"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("BLEU", f"{all_metrics.get('bleu_score', 0.0):.3f}")
                                st.metric("ROUGE-L", f"{all_metrics.get('rouge_l_score', 0.0):.3f}")
                            with col2:
                                st.metric("Tokens", f"{res.get('tokens_used', 0):,}")
                                st.metric("Cost (USD)", f"${res.get('cost_usd', 0.0):.6f}")
                            with col3:
                                st.metric("Gen Latency", f"{res.get('generation_latency', 0.0):.2f}s")
                                st.metric("Exec Latency", f"{res.get('execution_latency', 0.0):.2f}s")
                            
                            # Semantic match details
                            if all_metrics.get("semantic_details"):
                                st.text("Semantic Match Details:")
                                st.json(all_metrics["semantic_details"])
                    except Exception as e:
                        st.warning(f"Evaluation error: {e}")
                
                st.markdown("---")
        
        # Fallback handling if all failed
        if all(not r["success"] for r in results.values()):
            st.markdown("### 🔧 Fallback Options")
            
            # Expand schema
            with st.expander("📋 Expanded Schema (with sample values)"):
                try:
                    expanded_schema = expand_schema_snippet(engine, peek_values=True)
                    st.code(expanded_schema)
                except:
                    st.error("Could not expand schema")
            
            st.info("💡 **Suggestions:**\n"
                   "- Check that your question references valid columns\n"
                   "- Verify filter values match the data\n"
                   "- Try rephrasing the question\n"
                   "- Review the schema snippet above")
            
            # Show last errors
            st.markdown("### ❌ Last Errors")
            for technique, res in results.items():
                if res["error"]:
                    st.text(f"{technique}: {res['error']}")
        
        # Save to history
        st.session_state.query_history.append({
            "question": question,
            "results": results,
            "best": best_technique
        })
        
        # Save detailed results if evaluation mode is enabled
        if eval_mode and gold_sql and gold_sql.strip():
            # Collect all metrics for each technique
            saved_entry = {
                "question": question,
                "model_name": model_name,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gold_sql": gold_sql,
                "techniques": {}
            }
            
            for technique, res in results.items():
                technique_data = {
                    "success": res["success"],
                    "sql": res["sql"],
                    "latency": res["latency"],
                    "generation_latency": res.get("generation_latency", 0.0),
                    "execution_latency": res.get("execution_latency", 0.0),
                    "tokens_used": res.get("tokens_used", 0),
                    "input_tokens": res.get("input_tokens", 0),
                    "output_tokens": res.get("output_tokens", 0),
                    "cost_usd": res.get("cost_usd", 0.0),
                    "score": res["score"],
                    "refine_attempts": res.get("refine_attempts", 0)
                }
                
                # Add evaluation metrics if available
                if res["sql"]:
                    try:
                        all_metrics = calculate_all_metrics(
                            gold_sql,
                            res["sql"],
                            engine,
                            include_execution=True
                        )
                        technique_data.update({
                            "em": all_metrics.get("em", False),
                            "execution_accuracy": all_metrics.get("execution_accuracy", False),
                            "semantic_match": all_metrics.get("semantic_match", False),
                            "f1_score": all_metrics.get("f1_score", 0.0),
                            "bleu_score": all_metrics.get("bleu_score", 0.0),
                            "rouge_l_score": all_metrics.get("rouge_l_score", 0.0)
                        })
                    except Exception:
                        pass
                
                saved_entry["techniques"][technique] = technique_data
            
            st.session_state.saved_results.append(saved_entry)
            st.success(f"✅ Results saved! Total saved queries: {len(st.session_state.saved_results)}")
    
    # Query history
    if st.session_state.query_history:
        with st.expander("📜 Query History"):
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
                st.text(f"{i+1}. {entry['question']} → Best: {entry['best']}")
                if st.button(f"Re-run #{len(st.session_state.query_history) - i}", key=f"rerun_{i}"):
                    st.session_state.question_input = entry["question"]
                    st.rerun()


if __name__ == "__main__":
    main()

