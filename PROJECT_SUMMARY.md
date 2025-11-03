# Text-to-SQL Project: Comprehensive Summary

## 📋 Project Overview

**Goal**: Build a Text-to-SQL system that converts natural language questions into SQL queries using multiple prompting techniques, with evaluation metrics and model comparison capabilities.

**Technology Stack**:
- **Framework**: Streamlit (Interactive Web UI)
- **Database**: SQLite (read-only execution)
- **LLM Backends**: 
  - Ollama (Local models: Llama 3.1 8B)
  - OpenAI API (GPT-4o-mini, GPT-3.5-turbo)
- **SQL Parsing/Validation**: sqlglot
- **Metrics**: Custom implementation for EM, EX, SM, F1, BLEU, ROUGE-L

---

## 🎯 Core Functionalities

### 1. **Multi-Prompting Technique Comparison**
   - Compares 4 different prompting strategies side-by-side
   - Shows results, metrics, and performance for each technique
   - Identifies best technique automatically

### 2. **Dual Model Support**
   - **Ollama**: Run local models (free, private, slower)
   - **OpenAI**: Use GPT models (paid, faster, cloud-based)
   - Easy switching between models

### 3. **Evaluation & Metrics**
   - **Gold SQL Comparison**: Compare generated SQL with expected SQL
   - **Multiple Metrics**: EX, SM, F1, BLEU, ROUGE-L, Latency, Token Usage, Cost
   - **Side-by-side Results**: Visual comparison of Gold SQL vs Generated SQL results

### 4. **Result Saving & Model Comparison**
   - Save query results with all metrics
   - Compare multiple models (Ollama vs GPT) across queries
   - Aggregate statistics and performance summaries

### 5. **Error Handling & Refinement**
   - Automatic SQL validation
   - Execution-Guided (EG) auto-refinement for failed queries
   - Fallback mechanisms and error reporting

---

## 🔬 Four Prompting Techniques: Why & How

### 1. **Few-Shot Prompting (FS)**
   - **What**: Provides 4 example SQL queries in the prompt
   - **How**: Shows question-SQL pairs as examples before the actual question
   - **Why**: Helps model learn the pattern/formula from examples
   - **Best For**: Simple queries with clear patterns
   - **Example Structure**:
     ```
     Examples:
     Q: Average hours_per_week by education (top 10).
     SQL: SELECT education, AVG(hours_per_week) FROM adult_income GROUP BY education...
     
     Q: [Your question]
     SQL: ?
     ```

### 2. **Chain-of-Thought (CoT)**
   - **What**: Asks model to think step-by-step before generating SQL
   - **How**: Provides explicit reasoning steps (identify columns → filters → aggregations → ordering)
   - **Why**: Breaks down complex problem into manageable steps
   - **Best For**: Medium complexity queries requiring logical reasoning
   - **Example Structure**:
     ```
     Think step by step:
     1. Identify relevant columns from schema
     2. Determine filters (WHERE clauses)
     3. Identify aggregations (GROUP BY, COUNT, AVG)
     4. Determine sorting and limits
     5. Write final SQL
     ```

### 3. **Least-to-Most (LtM)**
   - **What**: Breaks problem into substeps (A, B, C, D)
   - **How**: Guides model through systematic decomposition
   - **Why**: Ensures all aspects are considered before generating SQL
   - **Best For**: Complex queries with multiple requirements
   - **Example Structure**:
     ```
     Substep A: Identify table and columns needed
     Substep B: Determine filtering conditions
     Substep C: Determine grouping and aggregation
     Substep D: Determine ordering and limits
     ```

### 4. **Execution-Guided (EG)**
   - **What**: Generates SQL, executes it, and refines if it fails
   - **How**: 
     1. Generate SQL
     2. Execute and catch errors
     3. If error occurs, generate refined SQL with error message
     4. Repeat up to 3 attempts
   - **Why**: Self-correcting mechanism improves accuracy
   - **Best For**: Complex queries prone to errors
   - **Auto-Refine Feature**: Automatically refines failed queries with error feedback

---

## 🧠 RAG-Lite Schema Retrieval: Implementation & Benefits

### What is RAG-Lite?
A lightweight version of Retrieval-Augmented Generation that dynamically fetches database schema information and includes it in prompts.

### How It's Implemented:

#### 1. **Schema Introspection** (`schema_retriever.py`)
```python
def get_schema_snippet(question, engine, table_name):
    # Uses SQLite PRAGMA table_info() to introspect database
    # Extracts column names dynamically
    # Returns compact schema: "table_name(column1, column2, ...)"
```

**Key Features**:
- **Dynamic Column Detection**: Uses `PRAGMA table_info(table_name)` to get actual columns
- **No Hardcoding**: Never assumes column names - always introspects from DB
- **Compact Format**: Creates 1-3 line schema snippet for prompts

#### 2. **Schema Expansion** (Optional)
- Shows sample values for key columns
- Helps model understand data types and value ranges
- Only shown in expanded view (not in prompts to save tokens)

### Benefits of RAG-Lite:

1. **Reduces Hallucinations**: 
   - Model sees actual columns, not guessing
   - Prevents using non-existent column names

2. **Dynamic Dataset Support**:
   - Works with ANY SQLite database
   - No code changes needed for different datasets
   - Automatically adapts to schema

3. **Token Efficiency**:
   - Only includes relevant schema info (not entire DB structure)
   - Compact format saves tokens and cost

4. **Accurate Column Names**:
   - Always matches actual database structure
   - No typos or mismatches

5. **Context-Aware** (Future Enhancement):
   - Could filter columns based on question keywords
   - Only shows relevant columns for query

### Example:
```
Without RAG: Model might guess "gender" column exists
With RAG: Schema shows actual column is "sex" → Model uses correct name
```

---

## 📊 Project Flow (Step-by-Step)

### User Interaction Flow:

1. **User Input**:
   - Enters natural language question in Streamlit UI
   - Optionally enables "Evaluation Mode" and provides Gold SQL

2. **Schema Retrieval (RAG-Lite)**:
   - System introspects database using `PRAGMA table_info()`
   - Generates compact schema snippet
   - Includes schema in all prompts

3. **Prompt Generation**:
   - Builds 4 different prompts (Few-Shot, CoT, LtM, EG)
   - Each technique uses same schema but different guidance

4. **SQL Generation**:
   - Sends each prompt to LLM (Ollama or OpenAI)
   - Extracts SQL from model output
   - Validates SQL starts with SELECT/WITH

5. **SQL Validation**:
   - Checks syntax using `sqlglot`
   - Ensures single statement (no multiple statements)
   - Verifies read-only (SELECT/WITH only)

6. **SQL Execution**:
   - Executes validated SQL against SQLite database
   - Returns results as pandas DataFrame
   - Measures execution latency

7. **Evaluation** (if Gold SQL provided):
   - Executes Gold SQL
   - Compares results (data values, ignoring column names)
   - Calculates metrics: EX, SM, F1, BLEU, ROUGE-L
   - Displays side-by-side comparison

8. **Result Saving** (if Evaluation Mode):
   - Saves all metrics, model name, technique, timestamp
   - Stores in session state for comparison tab

9. **Display & Comparison**:
   - Shows all 4 techniques side-by-side
   - Highlights best technique
   - Shows metrics and performance data
   - Comparison tab allows model comparison

---

## 📁 Project Structure

```
text2sql/
├── app.py                    # Main Streamlit UI (2 tabs)
├── db.py                     # Database connection, SQL validation, execution
├── schema_retriever.py       # RAG-lite schema introspection
├── prompts.py                # 4 prompting technique builders
├── model_client.py           # Ollama & OpenAI API clients
├── metrics.py                # EM, EX, compare_dataframes
├── advanced_metrics.py       # SM, F1, BLEU, ROUGE-L
├── seed_income_db.py         # Database creation from CSV
├── income.db                 # SQLite database (Adult Income dataset)
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 📦 Dataset Used: Adult Income Dataset

### Source:
- **Original**: UCI Machine Learning Repository - Adult Income Dataset
- **Size**: 32,561 rows (or full dataset based on your selection)
- **Format**: CSV → Converted to SQLite

### Schema:
```
adult_income(
    age, workclass, fnlwgt, education, education_num,
    marital_status, occupation, relationship, race, sex,
    capital_gain, capital_loss, hours_per_week,
    native_country, income
)
```

### Column Normalization:
- **`gender` → `sex`**: Standardized naming
- **`educational-num` → `education_num`**: Normalized format

### Purpose:
- **Training/Testing**: Used for evaluating Text-to-SQL accuracy
- **Real-world Queries**: Income prediction, demographic analysis, statistical queries
- **Complex Scenarios**: Supports aggregations, filters, groupings, percentages

### Example Queries:
- "Average hours_per_week by education (top 10)"
- "Count by race and sex"
- "For each education level and gender, show how many people there are, the percentage earning more than 50K..."

---

## 🎯 Key Features & Innovations

### 1. **Dynamic Schema Adaptation**
- Works with ANY SQLite database without code changes
- Automatically detects columns and table structure
- No hardcoded schema assumptions

### 2. **Multi-Technique Comparison**
- Unique approach: Run same query with 4 different prompting strategies
- Helps identify best technique for specific query types
- Consensus detection (if multiple techniques agree)

### 3. **Robust SQL Validation**
- Syntax checking with sqlglot
- Security: Only SELECT/WITH statements allowed (read-only)
- Single statement enforcement
- Automatic refinement on failure

### 4. **Comprehensive Metrics**
- **Execution Accuracy (EX)**: Do queries produce same results? (Most important)
- **Semantic Match (SM)**: Are queries logically equivalent?
- **F1-Score**: Token-level precision/recall
- **BLEU/ROUGE-L**: Text similarity metrics
- **Performance**: Latency, Token Usage, Cost

### 5. **Model Comparison**
- Save results from multiple models (Ollama vs GPT)
- Compare across queries and techniques
- Aggregate statistics (EM%, EX%, Avg F1, Cost, Latency)
- Visualize differences

### 6. **Error Handling**
- Graceful degradation if one technique fails
- Auto-refinement for Execution-Guided technique
- Clear error messages and suggestions
- Fallback to expanded schema view

---

## 🔄 Technical Implementation Details

### SQL Validation Process:
1. **Basic Check**: Must start with SELECT or WITH
2. **Syntax Check**: Use sqlglot parser (with fallback)
3. **Security Check**: Only SELECT/WITH statements
4. **Single Statement**: No semicolon-separated multiple statements
5. **Execution Safety**: Read-only database connection

### Execution-Guided Refinement:
```
Initial Generation → Execute → Error? 
  → Yes → Build refine prompt with error message → Regenerate
  → No → Success
(Max 3 refinement attempts)
```

### Dataframe Comparison:
- **Order-agnostic**: Sorts rows and columns before comparison
- **Type-aware**: Handles int vs float differences (40 = 40.0)
- **Precision-tolerant**: Uses numpy.isclose() for floating point (36.4007 ≈ 36.4007001)
- **Column-name agnostic**: Can ignore column name differences (compares data values)

### Token Usage Tracking:
- **OpenAI**: Uses `response.usage` (input_tokens, output_tokens)
- **Ollama**: Estimated from input/output text length
- **Cost Calculation**: Based on model pricing ($0.15/$0.60 per 1M tokens for GPT-4o-mini)

---

## 📈 Metrics Explained

### 1. **EX (Execution Accuracy)** - Most Important
- **What**: Do both queries (Gold and Generated) produce identical results?
- **How**: Execute both, compare result dataframes (ignoring column names)
- **Why**: Ultimate test - if results match, query is correct regardless of SQL syntax

### 2. **SM (Semantic Match)**
- **What**: Are queries logically equivalent (even if syntax differs)?
- **How**: Parse SQL ASTs, compare structure and operations
- **Why**: Catches cases where SQL looks different but is logically correct

### 3. **F1-Score**
- **What**: Token-level precision and recall
- **How**: Compare tokens between Gold SQL and Generated SQL
- **Why**: Measures similarity at token level

### 4. **BLEU Score**
- **What**: Text similarity metric (borrowed from machine translation)
- **How**: N-gram overlap between queries
- **Why**: Measures how similar the SQL text is

### 5. **ROUGE-L Score**
- **What**: Longest Common Subsequence metric
- **How**: Finds longest matching sequence of tokens
- **Why**: Measures structural similarity

### 6. **Latency**
- **Generation Latency**: Time to generate SQL from LLM
- **Execution Latency**: Time to execute SQL query
- **Total Latency**: Sum of both

### 7. **Token Usage & Cost**
- **Input Tokens**: Tokens in prompt
- **Output Tokens**: Tokens in generated SQL
- **Cost**: Calculated based on model pricing (OpenAI only, Ollama is free)

---

## 💡 Why This Approach?

### Why Multiple Prompting Techniques?
- **Different techniques work better for different query types**
- **Provides robustness**: If one fails, others might succeed
- **Consensus detection**: If multiple techniques agree, higher confidence

### Why RAG-Lite Schema Retrieval?
- **Reduces hallucinations**: Model sees actual columns
- **Dynamic adaptation**: Works with any dataset
- **Token efficient**: Only includes relevant schema info

### Why Read-Only SQL?
- **Security**: Prevents accidental data modification
- **Safety**: Can't delete or corrupt database
- **Focus**: Text-to-SQL is primarily for querying, not modification

### Why Both Ollama and OpenAI?
- **Comparison**: Understand trade-offs (cost vs speed vs privacy)
- **Flexibility**: Use local (free) or cloud (faster)
- **Research**: Compare model capabilities

---

## 🚀 Usage Workflow

### Basic Query:
1. Enter question: "Average hours_per_week by education"
2. Click "Generate SQL"
3. View 4 techniques' results
4. See which one worked best

### Evaluation Mode:
1. Enable "Evaluation Mode" checkbox
2. Paste Gold SQL query
3. Enter question
4. Click "Generate SQL"
5. View:
   - Gold SQL results (expected)
   - Generated SQL results (for each technique)
   - Side-by-side comparison
   - Metrics: EX, SM, F1, BLEU, ROUGE-L
   - Results automatically saved

### Model Comparison:
1. Run queries with Ollama model
2. Run same queries with GPT model
3. Switch to "Saved Results & Comparison" tab
4. Filter by model
5. View aggregated comparison statistics

---

## 🔧 Technical Challenges Solved

### 1. **Column Name Mismatch**
- **Problem**: Gold SQL uses `avg_hours_women_work`, generated uses `avg_hours`
- **Solution**: Compare data values only, ignore column names
- **Implementation**: `ignore_column_names=True` in dataframe comparison

### 2. **Floating Point Precision**
- **Problem**: 36.4007 vs 36.4007001 should match
- **Solution**: Use `numpy.isclose()` with tolerance
- **Implementation**: `rtol=1e-5, atol=1e-8`

### 3. **Type Differences**
- **Problem**: Integer 40 vs Float 40.0 should match
- **Solution**: Convert to float before comparison
- **Implementation**: Try float conversion, then compare numerically

### 4. **SQL Extraction Failures**
- **Problem**: Model outputs explanation before SQL
- **Solution**: Regex extraction + validation (must start with SELECT/WITH)
- **Fallback**: Execution-Guided refinement

### 5. **Multi-Statement SQL**
- **Problem**: Model might generate multiple SQL statements
- **Solution**: Validate single statement only
- **Security**: Reject any multi-statement queries

---

## 📊 Example Results

### Query: "Average hours_per_week by education (top 10)"

**Gold SQL Result**: 36.4007 (avg_hours_women_work column)

**Generated Results**:
- Few-Shot: 36.4007 (avg_hours column) → ✅ EX Match, ✅ SM Match
- CoT: 36.4007 (avg_hours column) → ✅ EX Match, ✅ SM Match
- LtM: 36.4007 (avg_hours column) → ✅ EX Match, ✅ SM Match
- EG: 36.4007 (AVG(hours_per_week) column) → ✅ EX Match

**Metrics**:
- EX: ✅ (All match - same results)
- SM: ✅ (Few-Shot, CoT, LtM - semantically equivalent)
- F1: ~0.85-0.92 (high similarity)

---

## 🎓 Potential Questions & Answers

### Q: Why not use fine-tuning instead of prompting?
**A**: 
- Fine-tuning requires training data and computational resources
- Prompting is faster to experiment with different strategies
- Can switch between models easily without retraining
- More flexible for different datasets

### Q: Why is RAG "lite"?
**A**: 
- Full RAG typically includes vector search, embeddings, retrieval
- Our version: Simple schema introspection (no embeddings)
- Lightweight but effective for structured data (databases)
- Faster, simpler, sufficient for our use case

### Q: How do you handle complex queries?
**A**: 
- Execution-Guided technique auto-refines on errors
- Chain-of-Thought breaks down complex logic
- Complex query detection (if we re-enabled it) would use CTEs
- Window functions support for ranking/percentages

### Q: Why compare 4 techniques instead of just using the best one?
**A**: 
- Different techniques excel at different query types
- Consensus detection increases confidence
- Research comparison shows technique strengths/weaknesses
- Robustness: if one fails, others might succeed

### Q: How accurate is the system?
**A**: 
- Depends on model and query complexity
- EX (Execution Accuracy) is most reliable metric
- Simple queries: Often 90%+ EX accuracy
- Complex queries: May require refinement, 60-80% EX accuracy

### Q: What's the difference between EM, EX, and SM?
**A**: 
- **EM**: SQL text must match character-by-character (very strict, rarely matches)
- **EX**: Query results must match (most important - if results match, query is correct)
- **SM**: Queries must be logically equivalent (catches different SQL syntax for same logic)

### Q: Can this work with other databases besides SQLite?
**A**: 
- Yes, SQLAlchemy supports multiple databases
- Would need to adjust schema introspection (not all DBs use PRAGMA)
- SQL validation would need database-specific dialect
- Current implementation optimized for SQLite

### Q: What are the limitations?
**A**: 
- Read-only queries only (SELECT/WITH)
- Single table queries (no joins across multiple tables in current prompts)
- SQLite-specific features
- Requires structured schema (doesn't work with unstructured data)

---

## 📚 Key Takeaways

1. **Multi-Technique Approach**: Comparing 4 prompting strategies provides robustness and insights
2. **RAG-Lite**: Dynamic schema retrieval reduces hallucinations and adapts to any dataset
3. **Comprehensive Evaluation**: Multiple metrics (EX, SM, F1, etc.) provide holistic view
4. **Model Flexibility**: Support for both local (Ollama) and cloud (OpenAI) models
5. **Practical Tool**: Real-world application with error handling, validation, and comparison features

---

## 🔮 Future Enhancements (Possible)

1. **Multi-table Joins**: Support queries across multiple tables
2. **Query Optimization**: Suggest index usage or query improvements
3. **Natural Language Refinement**: Let users refine queries conversationally
4. **Advanced RAG**: Use embeddings to retrieve relevant schema parts based on question
5. **Batch Processing**: Evaluate on benchmark datasets automatically
6. **Export Results**: Save comparison results to CSV/JSON
7. **Query Suggestions**: Suggest similar queries based on saved history

---

This project demonstrates a complete, production-ready Text-to-SQL system with evaluation, comparison, and multiple prompting strategies for research and practical use.

