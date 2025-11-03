"""Streamlit app for comparing GPT vs Ollama models side-by-side."""
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from db import get_engine
from model_comparison import compare_models, batch_evaluate, get_memory_usage
from advanced_metrics import calculate_all_metrics

load_dotenv()

st.set_page_config(
    page_title="Model Comparison: GPT vs Ollama",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ GPT vs Ollama Model Comparison")
st.markdown("Compare GPT and Ollama models across all metrics side-by-side.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Models to Compare")
    use_gpt = st.checkbox("GPT Model", value=True)
    gpt_model = st.text_input(
        "GPT Model Name",
        value=os.getenv("MODEL_NAME_GPT", "openai/gpt-4o-mini"),
        help="Format: openai/model-name"
    ) if use_gpt else None
    
    use_ollama = st.checkbox("Ollama Model", value=True)
    ollama_model = st.text_input(
        "Ollama Model Name",
        value=os.getenv("MODEL_NAME_OLLAMA", "ollama/llama3.1:8b"),
        help="Format: ollama/model-name"
    ) if use_ollama else None
    
    # Technique selection
    st.subheader("Techniques")
    techniques = []
    if st.checkbox("Few-Shot", value=True):
        techniques.append("Few-Shot")
    if st.checkbox("CoT", value=True):
        techniques.append("CoT")
    if st.checkbox("LtM", value=True):
        techniques.append("LtM")
    if st.checkbox("EG", value=True):
        techniques.append("EG")
    
    # Database
    db_url = st.text_input(
        "Database URL",
        value=os.getenv("DB_URL", "sqlite:///income.db"),
        help="SQLite path or connection string"
    )

# Main comparison interface
tab1, tab2 = st.tabs(["Single Query Comparison", "Batch Evaluation"])

with tab1:
    st.header("Single Query Comparison")
    
    question = st.text_input(
        "💬 Natural Language Question",
        placeholder="e.g., Average hours_per_week by education (top 10)"
    )
    
    gold_sql = st.text_area(
        "📝 Gold SQL (Required for metrics)",
        height=100,
        placeholder="Paste the correct SQL query for this question"
    )
    
    if st.button("🚀 Compare Models", type="primary"):
        if not question:
            st.error("Please enter a question")
        elif not gold_sql:
            st.warning("⚠️ Gold SQL is required for accurate comparison. Metrics will be limited.")
        else:
            models = []
            if use_gpt and gpt_model:
                models.append(gpt_model)
            if use_ollama and ollama_model:
                models.append(ollama_model)
            
            if not models:
                st.error("Please select at least one model")
            elif not techniques:
                st.error("Please select at least one technique")
            else:
                try:
                    with st.spinner("Comparing models..."):
                        results = compare_models(
                            question,
                            gold_sql,
                            models,
                            techniques,
                            db_url
                        )
                    
                    # Display comparison results
                    for model in models:
                        st.header(f"📊 {model}")
                        
                        for technique in techniques:
                            if technique not in results.get(model, {}):
                                continue
                            
                            res = results[model][technique]
                            
                            with st.expander(f"{technique} - {model}", expanded=True):
                                # Basic metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    status = "✅ Success" if res["success"] else "❌ Failed"
                                    st.metric("Status", status)
                                
                                with col2:
                                    st.metric("Latency", f"{res['total_latency']:.2f}s")
                                
                                with col3:
                                    st.metric("Tokens", f"{res['tokens_used']:,}")
                                
                                with col4:
                                    cost = res.get("cost_usd", 0.0)
                                    st.metric("Cost", f"${cost:.6f}" if cost > 0 else "Free")
                                
                                # SQL
                                st.subheader("Generated SQL")
                                if res["sql"]:
                                    st.code(res["sql"], language="sql")
                                else:
                                    st.error(f"No SQL generated. Error: {res.get('error', 'Unknown')}")
                                
                                # Results
                                if res["success"]:
                                    st.subheader("Query Results")
                                    if res.get("df") is not None:
                                        st.dataframe(res["df"], use_container_width=True)
                                
                                # All metrics
                                if gold_sql and res["sql"]:
                                    st.subheader("📊 Comprehensive Metrics")
                                    
                                    metrics = res.get("metrics", {})
                                    
                                    # Core metrics
                                    metrics_cols = st.columns(5)
                                    with metrics_cols[0]:
                                        st.metric("EM", "✅" if metrics.get("em") else "❌")
                                    with metrics_cols[1]:
                                        st.metric("EX", "✅" if metrics.get("execution_accuracy") else "❌")
                                    with metrics_cols[2]:
                                        st.metric("SM", "✅" if metrics.get("semantic_match") else "❌")
                                    with metrics_cols[3]:
                                        st.metric("F1", f"{metrics.get('f1_score', 0.0):.3f}")
                                    with metrics_cols[4]:
                                        st.metric("BLEU", f"{metrics.get('bleu_score', 0.0):.3f}")
                                    
                                    # Detailed metrics
                                    with st.expander("📈 Detailed Metrics"):
                                        detail_cols = st.columns(3)
                                        with detail_cols[0]:
                                            st.metric("ROUGE-L", f"{metrics.get('rouge_l_score', 0.0):.3f}")
                                            if metrics.get("f1_details"):
                                                st.json(metrics["f1_details"])
                                        with detail_cols[1]:
                                            if metrics.get("bleu_details"):
                                                st.json(metrics["bleu_details"])
                                        with detail_cols[2]:
                                            if metrics.get("semantic_details"):
                                                st.json(metrics["semantic_details"])
                                
                                # Memory usage
                                if res.get("memory_delta_mb"):
                                    st.caption(f"Memory Delta: CPU {res['memory_delta_mb'].get('cpu_memory_mb', 0):.1f} MB, "
                                              f"GPU {res['memory_delta_mb'].get('gpu_memory_mb', 0):.1f} MB")
                                
                                st.markdown("---")
                    
                    # Summary comparison table
                    st.header("📋 Summary Comparison")
                    summary_data = []
                    
                    for model in models:
                        for technique in techniques:
                            if technique not in results.get(model, {}):
                                continue
                            res = results[model][technique]
                            metrics = res.get("metrics", {})
                            
                            summary_data.append({
                                "Model": model,
                                "Technique": technique,
                                "Success": "✅" if res["success"] else "❌",
                                "EM": "✅" if metrics.get("em") else "❌",
                                "EX": "✅" if metrics.get("execution_accuracy") else "❌",
                                "SM": "✅" if metrics.get("semantic_match") else "❌",
                                "F1": f"{metrics.get('f1_score', 0.0):.3f}",
                                "BLEU": f"{metrics.get('bleu_score', 0.0):.3f}",
                                "ROUGE-L": f"{metrics.get('rouge_l_score', 0.0):.3f}",
                                "Latency (s)": f"{res['total_latency']:.2f}",
                                "Tokens": f"{res['tokens_used']:,}",
                                "Cost (USD)": f"${res.get('cost_usd', 0.0):.6f}" if res.get('cost_usd', 0.0) > 0 else "Free"
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Comparison error: {str(e)}")

with tab2:
    st.header("Batch Evaluation")
    st.markdown("Upload a CSV file with test cases or enter multiple questions manually.")
    
    upload_method = st.radio(
        "Input Method",
        ["CSV Upload", "Manual Entry"]
    )
    
    test_cases = []
    
    if upload_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "question" in df.columns and "gold_sql" in df.columns:
                test_cases = df[["question", "gold_sql"]].to_dict("records")
                st.success(f"Loaded {len(test_cases)} test cases")
                st.dataframe(df.head())
            else:
                st.error("CSV must have 'question' and 'gold_sql' columns")
    
    else:
        num_cases = st.number_input("Number of test cases", min_value=1, max_value=20, value=3)
        for i in range(num_cases):
            with st.expander(f"Test Case {i+1}"):
                question = st.text_input(f"Question {i+1}", key=f"q_{i}")
                gold_sql = st.text_area(f"Gold SQL {i+1}", key=f"gsql_{i}", height=80)
                if question and gold_sql:
                    test_cases.append({"question": question, "gold_sql": gold_sql})
    
    if test_cases and st.button("🚀 Run Batch Evaluation"):
        models = []
        if use_gpt and gpt_model:
            models.append(gpt_model)
        if use_ollama and ollama_model:
            models.append(ollama_model)
        
        if not models:
            st.error("Please select at least one model")
        elif not techniques:
            st.error("Please select at least one technique")
        else:
            try:
                with st.spinner("Running batch evaluation (this may take a while)..."):
                    batch_results = batch_evaluate(
                        test_cases,
                        models,
                        techniques,
                        db_url
                    )
                
                # Display aggregated results
                st.header("📊 Aggregated Results")
                
                aggregated = batch_results.get("aggregated", {})
                
                for model in models:
                    st.subheader(f"📈 {model}")
                    
                    model_data = []
                    for technique in techniques:
                        if technique in aggregated.get(model, {}):
                            agg = aggregated[model][technique]
                            model_data.append({
                                "Technique": technique,
                                "EM Avg": f"{agg['em_avg']:.3f}",
                                "EX Avg": f"{agg['ex_avg']:.3f}",
                                "F1 Avg": f"{agg['f1_avg']:.3f}",
                                "BLEU Avg": f"{agg['bleu_avg']:.3f}",
                                "ROUGE-L Avg": f"{agg['rouge_l_avg']:.3f}",
                                "Success Rate": f"{agg['success_rate']:.3f}",
                                "Avg Latency (s)": f"{agg['latency_avg']:.2f}",
                                "Total Cost": f"${agg['cost_total']:.6f}" if agg['cost_total'] > 0 else "Free",
                                "Tests": int(agg['num_tests'])
                            })
                    
                    if model_data:
                        st.dataframe(pd.DataFrame(model_data), use_container_width=True)
                
                # Export results
                st.download_button(
                    "📥 Download Results (JSON)",
                    data=str(batch_results),
                    file_name="comparison_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Batch evaluation error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

