# Text-to-SQL: Multi-Technique Prompting

A clean, minimal Text-to-SQL project that compares four prompting techniques (Few-Shot, Chain-of-Thought, Least-to-Most, Execution-Guided) with RAG-lite schema retrieval and evaluation metrics.

## Features

- **Four Prompting Techniques** running in parallel:
  - **Few-Shot (FS)**: Examples-based learning
  - **Chain-of-Thought (CoT)**: Step-by-step reasoning
  - **Least-to-Most (LtM)**: Decomposed problem-solving
  - **Execution-Guided (EG)**: Generate → execute → auto-refine (≤2 attempts)

- **RAG-lite Schema Retrieval**: Compact schema snippets to reduce hallucinations
- **SQL Validation**: SELECT/WITH-only, sqlglot parsing
- **Read-only Execution**: SQLAlchemy against SQLite
- **Evaluation Mode**: Exact Match (EM) and Execution Accuracy (EX) metrics
- **Multi-backend Support**: Ollama (local) and OpenAI

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Database

Place your `adult.csv` file in the project directory (or in `adult/adult.csv`), then:

```bash
python seed_income_db.py
```

This creates `income.db` with the first 2,000 rows from the CSV.

### 3. Setup Model (Ollama)

For local models via Ollama:

```bash
ollama pull llama3.1:8b
```

Or try other models:
```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

### 4. Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:
```
DB_URL=sqlite:///income.db
MODEL_NAME=ollama/llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

For OpenAI, use:
```
MODEL_NAME=openai/gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the App

```bash
streamlit run app.py
```

## Usage

### Basic Query

1. Enter a natural language question, e.g.:
   - "Average hours_per_week by education (top 10)"
   - "Top 5 occupation by count where income = '>50K'"
   - "Count by race and sex"

2. Click "🚀 Generate SQL"

3. View results for all four techniques:
   - SQL code block
   - Result table (or error)
   - Latency
   - Score
   - Best technique banner

### Evaluation Mode

1. Enable "Evaluation Mode" in the sidebar
2. Enter your question
3. Paste the correct "Gold SQL" in the textarea
4. Click "Generate SQL"
5. View EM (Exact Match) and EX (Execution Accuracy) badges for each technique

### Consensus Detection

When 2+ techniques return identical results, a confidence pill appears: "✅ Higher confidence (consensus)".

## Project Structure

```
text2sql/
├── app.py                 # Streamlit UI (main entry point)
├── model_client.py        # Ollama + OpenAI backends
├── prompts.py             # FS/CoT/LtM/EG prompt builders
├── db.py                  # SQLAlchemy, validation, execution
├── schema_retriever.py    # RAG-lite schema snippet
├── seed_income_db.py      # Build income.db from CSV
├── metrics.py             # EM/EX evaluation helpers
├── income.db              # SQLite database (created by seed script)
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Database Schema

The `adult_income` table contains:
- `age`, `workclass`, `fnlwgt`, `education`, `education_num`
- `marital_status`, `occupation`, `relationship`, `race`, `sex`
- `capital_gain`, `capital_loss`, `hours_per_week`
- `native_country`, `income` (values: `'<=50K'`, `'>50K'`)

## Testing Scenarios

### 1. Database Seeding
```bash
python seed_income_db.py
# Should create income.db with ≥1,000 rows
```

### 2. Basic Query
- Ask: "Average hours_per_week by education (top 10)"
- At least one technique should succeed

### 3. SQL Validation
- Try pasting a DELETE statement → should be rejected

### 4. EG Auto-Refine
- Enable EG Auto-Refine
- A broken query should auto-fix within 2 attempts

### 5. Evaluation Mode
- Enable Evaluation Mode
- Provide Gold SQL
- Check EM/EX badges

## Configuration

### Model Backends

**Ollama (Local)**:
- Format: `ollama/model-name`
- Example: `ollama/llama3.1:8b`
- Requires: Ollama running locally

**OpenAI**:
- Format: `openai/model-name`
- Example: `openai/gpt-4o-mini`
- Requires: `OPENAI_API_KEY` in `.env`

### Database

- Default: `sqlite:///income.db`
- Custom: Set `DB_URL` in `.env` (e.g., `sqlite:///path/to/db.db`)

## Troubleshooting

### Database Connection Error
- Ensure `income.db` exists (run `seed_income_db.py`)
- Check `DB_URL` in `.env`

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` in `.env`
- Verify model is downloaded: `ollama list`

### All Techniques Failing
- Check schema snippet matches your database
- Verify question references valid columns
- Try simpler queries first
- Review error messages in the UI

### OpenAI API Error
- Verify `OPENAI_API_KEY` is set
- Check API key validity
- Ensure you have credits/quota

## License

MIT License - feel free to use and modify.

## Contributing

Contributions welcome! Areas for improvement:
- Additional prompting techniques
- Better schema retrieval
- More robust error handling
- Additional database backends
- Performance optimizations

