# Text-to-SQL: Multi-Technique Prompting with Evaluation

A comprehensive Text-to-SQL system that compares four prompting techniques (Few-Shot, Chain-of-Thought, Least-to-Most, Execution-Guided) with RAG-lite schema retrieval, comprehensive evaluation metrics, and model comparison capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Features

- **Four Prompting Techniques**: Compare Few-Shot, Chain-of-Thought, Least-to-Most, and Execution-Guided side-by-side
- **RAG-Lite Schema Retrieval**: Dynamic schema introspection to reduce hallucinations
- **Dual Model Support**: Works with Ollama (local) and OpenAI (GPT) models
- **Comprehensive Evaluation**: EX, SM, F1, BLEU, ROUGE-L, Latency, Token Usage, Cost metrics
- **Model Comparison**: Save and compare results across multiple models
- **Interactive UI**: Clean Streamlit interface with two tabs (Query & Comparison)
- **Robust Error Handling**: SQL validation, auto-refinement, and graceful fallbacks

## 📋 Requirements

- Python 3.8+
- Ollama (for local models) OR OpenAI API key
- SQLite database

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deveshs18/Text-To-SQL.git
   cd Text-To-SQL
   ```

2. **Install dependencies**
   ```bash
   cd text2sql
   pip install -r requirements.txt
   ```

3. **Setup database**
   ```bash
   # Download Adult Income dataset CSV
   # Place it in the project directory or adult/ folder
   python seed_income_db.py
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

### Basic Query
1. Enter your natural language question
2. Click "Generate SQL"
3. View results from all four techniques

### Evaluation Mode
1. Enable "Evaluation Mode" checkbox
2. Paste Gold SQL (expected correct query)
3. Enter question and generate
4. View metrics: EX, SM, F1, BLEU, ROUGE-L, Cost, Latency

### Model Comparison
1. Run queries with different models
2. Switch to "Saved Results & Comparison" tab
3. Filter by model or technique
4. View aggregated comparison statistics

## 📁 Project Structure

```
text2sql/
├── app.py                 # Main Streamlit UI (2 tabs)
├── db.py                  # Database connection & SQL execution
├── schema_retriever.py    # RAG-lite schema introspection
├── prompts.py             # 4 prompting technique builders
├── model_client.py        # Ollama & OpenAI clients
├── metrics.py             # EM, EX evaluation
├── advanced_metrics.py    # SM, F1, BLEU, ROUGE-L
├── seed_income_db.py      # Database creation script
└── requirements.txt       # Dependencies
```

## 🔬 Prompting Techniques

1. **Few-Shot**: Provides example SQL queries as learning patterns
2. **Chain-of-Thought**: Step-by-step reasoning guidance
3. **Least-to-Most**: Problem decomposition into substeps
4. **Execution-Guided**: Generate → Execute → Auto-refine on errors

## 📊 Evaluation Metrics

- **EX (Execution Accuracy)**: Do queries produce same results? ⭐ Most important
- **SM (Semantic Match)**: Are queries logically equivalent?
- **F1-Score**: Token-level precision/recall
- **BLEU/ROUGE-L**: Text similarity metrics
- **Performance**: Latency, Token Usage, Cost

## 🧠 RAG-Lite Schema Retrieval

Dynamically introspects database schema using `PRAGMA table_info()` and includes compact schema snippets in prompts. This:
- Reduces hallucinations (model sees actual columns)
- Adapts to any SQLite database dynamically
- Saves tokens with compact format

## 📚 Dataset

Uses the **Adult Income Dataset** (UCI Machine Learning Repository):
- 32,561 rows
- 15 columns (age, education, occupation, income, etc.)
- Used for demographic and income prediction queries

## 📝 Documentation

See `PROJECT_SUMMARY.md` for comprehensive project documentation, including:
- Detailed explanations of all components
- Flow diagrams
- Technical challenges and solutions
- Q&A for presentations

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional prompting techniques
- Support for more database backends
- Enhanced schema retrieval
- Query optimization suggestions

## 📄 License

MIT License - feel free to use and modify.

## 👤 Author

**deveshs18**
- GitHub: [@deveshs18](https://github.com/deveshs18)

## 🙏 Acknowledgments

- Adult Income Dataset: UCI Machine Learning Repository
- Ollama for local LLM support
- OpenAI for GPT models
- Streamlit for the UI framework


