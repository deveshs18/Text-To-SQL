# Latency Comparison Summary

## Available Data

**Current Status:** Only GPT-4o-mini results are available. Ollama evaluation results are missing.

## GPT-4o-mini Latency Data

Based on 400 queries (100 test cases × 4 techniques):

### Overall Statistics:
- **Average Latency (With RAG):** 4.14 seconds
- **Average Latency (Without RAG):** 4.36 seconds
- **RAG Impact:** -0.22 seconds (5.3% faster with RAG)

### By Technique (With RAG):
- **Few-Shot:** ~1.5-2.0 seconds average
- **CoT:** ~3-5 seconds average (longer due to step-by-step reasoning)
- **LtM:** ~3-5 seconds average
- **EG:** ~2-4 seconds average (includes refinement time)

### By Technique (Without RAG):
- **Few-Shot:** ~0.6-1.2 seconds average
- **CoT:** ~2-4 seconds average
- **LtM:** ~2-4 seconds average
- **EG:** ~2-3 seconds average

## Missing Data

**Ollama (llama3.1:8b) Results:**
- ❌ No result files found
- Expected files: `ollama_llama3.1_8b_*_results.csv`
- Likely reason: Ollama evaluation didn't complete or was skipped

## Expected Ollama Latency (Based on Previous Runs)

From earlier test runs, typical Ollama latencies:
- **Average:** 10-15 seconds per query
- **Range:** 5-25 seconds depending on query complexity
- **GPT is typically 3-5x faster** than Ollama

## To Get Complete Comparison

1. **Re-run Ollama evaluation:**
   ```bash
   cd text2sql/scripts
   python generate_100_evaluation.py
   ```
   (Make sure Ollama is running and API key is not set, so it runs Ollama-only)

2. **Or run Ollama separately:**
   - Use `compare_rag.py` with model set to `ollama/llama3.1:8b`
   - Run for all 4 techniques
   - Results will be saved as `ollama_llama3.1_8b_*_results.csv`

## Generated Charts

✅ **latency_comparison.png** - Created but only shows GPT data (Ollama data missing)

The chart includes:
- Average latency by technique (With/Without RAG)
- Overall average comparison
- Speedup factor (when Ollama data is available)


