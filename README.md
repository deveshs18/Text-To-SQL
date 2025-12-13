# Text-to-SQL: Model Comparison Project

A comprehensive text-to-SQL system that compares two state-of-the-art models:
- **Arctic Fine-tuned Model** (Fine-tuned Arctic-Text2SQL-R1-7B with LoRA adapters)
- **GPT-4o-mini** (OpenAI's efficient language model)

## Features

- **Multiple Prompting Techniques**: Few-Shot, Chain-of-Thought (CoT), Least-to-Most (LtM), Execution-Guided (EG)
- **RAG-lite Schema Retrieval**: Automatic database schema extraction
- **Real-time SQL Generation**: Interactive Streamlit interface
- **Model Comparison**: Side-by-side comparison of both models
- **Evaluation Metrics**: Exact Match, Execution Accuracy, and more

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (for Arctic fine-tuned model - 8GB+ VRAM recommended)
- **OpenAI API Key** (for GPT-4o-mini)
- **Windows/Linux/MacOS**

## Important Notes

- **Fine-tuned Model Included**: The `arctic_lora_model/` directory contains the fine-tuned LoRA adapters. The base model will be downloaded automatically from Hugging Face when you first run the server.
- **No GGUF Model Needed**: The large GGUF model file (370MB+) is not included in this repository. It's not needed since we use the fine-tuned model directly via the server.
- **Database Setup**: You'll need to set up your database (see Database Setup section below).

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 266-TXT_SQL
```

### 2. Install Dependencies

```bash
pip install -r text2sql/requirements.txt
```

**Additional Requirements for Arctic Fine-tuned Model:**

If you plan to use the Arctic fine-tuned model, install these additional packages:

```bash
# PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers and related libraries
pip install transformers peft trl bitsandbytes accelerate datasets
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required for GPT-4o-mini)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=openai/gpt-4o-mini  # or ollama/arctic-finetuned
OLLAMA_BASE_URL=http://localhost:11437  # For Arctic fine-tuned model

# Database Configuration
DB_URL=sqlite:///income.db
```

## Database Setup

### Option 1: Use Existing Database

If you have an existing SQLite database, place it in the project root and update `DB_URL` in `.env`:

```bash
DB_URL=sqlite:///your_database.db
```

### Option 2: Set Up Adult Income Database (Default)

The project includes a script to set up the default `adult_income` database:

```bash
python text2sql/scripts/setup_database.py
```

This will:
- Create `income.db` in the project root
- Load data from `adult/adult.csv`
- Create the `adult_income` table with proper schema

### Option 3: Load Custom CSV to Database

To load your own CSV file:

```bash
python text2sql/scripts/load_csv_to_db.py
```

Follow the prompts to:
1. Enter your CSV file path
2. Enter table name
3. Enter database path (default: `income.db`)

## Running the Models

### Running GPT-4o-mini

**1. Set up environment:**

```bash
# In .env file
MODEL_NAME=openai/gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
```

**2. Start the Streamlit app:**

```bash
streamlit run text2sql/app.py
```

**3. Access the interface:**

Open your browser to `http://localhost:8501`

### Running Arctic Fine-tuned Model

**1. Start the Fine-tuned Model Server:**

Open a terminal and run:

```bash
python text2sql/scripts/finetuned_arctic_server.py
```

The server will:
- **Automatically download** the base Arctic model (Snowflake/Arctic-Text2SQL-R1-7B) from Hugging Face on first run (~14GB, but uses 4-bit quantization)
- Load your LoRA adapters from `arctic_lora_model/` (already included in repository)
- Start serving on `http://localhost:11437`

**Note**: The first time you run this, it will download the base model (~14GB). This is a one-time download. Subsequent runs will use the cached model.

**Expected output:**
```
================================================================================
LOADING FINE-TUNED ARCTIC MODEL
================================================================================
[1/3] Loading base model: Snowflake/Arctic-Text2SQL-R1-7B
✅ Base model loaded!

[2/3] Loading LoRA adapters from: arctic_lora_model
✅ LoRA adapters loaded!

[3/3] Model ready for inference!
================================================================================
SERVER STARTING
================================================================================
URL: http://localhost:11437
```

**2. Configure the app:**

Update `.env`:
```bash
MODEL_NAME=ollama/arctic-finetuned
OLLAMA_BASE_URL=http://localhost:11437
```

**3. Start the Streamlit app (in a new terminal):**

```bash
streamlit run text2sql/app.py
```

## Using the Application

### Main Interface

1. **Enter your question** in natural language (e.g., "What is the average age of people earning more than 50K?")

2. **Select model** in the sidebar:
   - `openai/gpt-4o-mini` for GPT-4o-mini
   - `ollama/arctic-finetuned` for Arctic fine-tuned model

3. **Click "Generate SQL"** to see results from all four prompting techniques:
   - **Few-Shot**: Examples-based prompting
   - **CoT**: Chain-of-Thought reasoning
   - **LtM**: Least-to-Most decomposition
   - **EG**: Execution-Guided refinement

### Features

- **Schema Display**: View the database schema used for generation
- **SQL Results**: See generated SQL and execution results
- **Metrics**: Compare Exact Match and Execution Accuracy
- **Query History**: Save and compare previous queries
- **Evaluation Mode**: Compare against gold-standard SQL

## Model Comparison

### Switching Between Models

**To compare models:**

1. **Test with GPT-4o-mini:**
   - Set `MODEL_NAME=openai/gpt-4o-mini` in `.env`
   - Restart Streamlit app
   - Run your queries

2. **Test with Arctic Fine-tuned:**
   - Ensure `finetuned_arctic_server.py` is running
   - Set `MODEL_NAME=ollama/arctic-finetuned` in `.env`
   - Set `OLLAMA_BASE_URL=http://localhost:11437` in `.env`
   - Restart Streamlit app
   - Run the same queries

3. **Compare Results:**
   - Use the "Saved Results" tab to compare performance
   - Check metrics (Exact Match, Execution Accuracy)
   - Compare SQL quality and correctness

## Project Structure

```
266-TXT_SQL/
├── text2sql/
│   ├── app.py                    # Main Streamlit application
│   ├── model_client.py           # Model API client (OpenAI/Ollama)
│   ├── prompts.py                # Prompt building functions
│   ├── db.py                     # Database utilities
│   ├── schema_retriever.py      # Schema extraction
│   ├── metrics.py                # Evaluation metrics
│   ├── advanced_metrics.py      # Advanced evaluation
│   ├── scripts/
│   │   ├── finetuned_arctic_server.py  # Arctic fine-tuned model server
│   │   ├── finetune_arctic.py          # Fine-tuning script
│   │   ├── setup_database.py           # Database setup
│   │   └── load_csv_to_db.py           # CSV loader
│   └── requirements.txt
├── arctic_lora_model/            # Fine-tuned LoRA adapters (included in repo)
├── models/                       # (Empty - large model files excluded via .gitignore)
├── adult/                        # Adult income dataset
├── income.db                     # SQLite database (created by setup script)
└── .env                          # Environment variables (create this file)

**Note**: The base Arctic model (~14GB) is NOT included. It will be automatically downloaded from Hugging Face when you first run `finetuned_arctic_server.py`.
```

## Troubleshooting

### Arctic Fine-tuned Model Issues

**Problem: Model server fails to start**

- **Check CUDA availability:**
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

- **Check if LoRA adapters exist:**
  ```bash
  ls arctic_lora_model/
  ```
  Should contain: `adapter_model.safetensors`, `adapter_config.json`

- **Memory issues:**
  - Ensure you have 8GB+ VRAM
  - Model uses 4-bit quantization automatically

**Problem: "Model not loaded" error**

- Ensure the server is running on port 11437
- Check `OLLAMA_BASE_URL` in `.env` matches the server port
- Verify server output shows "Model ready for inference!"

### GPT-4o-mini Issues

**Problem: "OpenAI API error"**

- Verify `OPENAI_API_KEY` is set in `.env`
- Check your OpenAI account has credits
- Ensure internet connection is active

**Problem: Rate limit errors**

- GPT-4o-mini has rate limits
- Wait a few seconds between requests
- Consider using async generation (already implemented)

### Database Issues

**Problem: "Table not found"**

- Run `python text2sql/scripts/setup_database.py`
- Verify `DB_URL` in `.env` points to correct database
- Check table name matches your schema

**Problem: Schema retrieval fails**

- Ensure database file exists and is accessible
- Check database permissions
- Verify table names are correct

## Fine-tuning Your Own Model

If you want to fine-tune the Arctic model or train a completely new model:

**📖 See detailed guide:** `TRAINING_AND_SWITCHING_MODELS.md`

**Quick steps:**

1. **Prepare your dataset** in Alpaca format (see `ft_spider/data/spider_alpaca_full.json` for example)

2. **Configure training** (edit `text2sql/scripts/finetune_arctic.py`):
   - `OUTPUT_DIR`: Directory name for your model adapters
   - `DATASET_PATH`: Path to your training dataset
   - `TRAINING_SUBSET_SIZE`: Number of examples (None for full dataset)

3. **Run fine-tuning:**
   ```bash
   python text2sql/scripts/finetune_arctic.py
   ```

4. **Create server for your model:**
   - Copy `finetuned_arctic_server.py` to `your_model_server.py`
   - Update `LORA_MODEL` and port number
   - Start server: `python text2sql/scripts/your_model_server.py`

5. **Switch models in Streamlit:**
   - Change `MODEL_NAME` in `.env` or Streamlit sidebar
   - Use `ollama/your-model-name` for your model
   - Use `openai/gpt-4o-mini` for GPT-4o-mini

**For complete instructions on training, server setup, and model switching, see `TRAINING_AND_SWITCHING_MODELS.md`**

## Running Qwen 0.5 Model

### Qwen 0.5 Finetuned Model

**1. Start the Finetuned Model Server:**

```bash
python text2sql/scripts/qwen_server.py
```

The server will:
- Load the base Qwen model (Qwen/Qwen2.5-Coder-0.5B) from Hugging Face on first run
- Load LoRA adapters from `qwen0p5b-spider-lora/` (included in repository)
- Start serving on `http://localhost:11438`

**2. Configure the app:**

Update `.env`:
```bash
MODEL_NAME=ollama/qwen-0.5b-spider
OLLAMA_BASE_URL=http://localhost:11438
```

### Qwen 0.5 Base Model (Baseline)

**1. Start the Base Model Server:**

```bash
python text2sql/scripts/qwen_base_server.py
```

The server will:
- Load the base Qwen model directly from Hugging Face (no LoRA adapters)
- Start serving on `http://localhost:11439`

**2. Configure the app:**

Update `.env`:
```bash
MODEL_NAME=ollama/qwen-0.5b-base
OLLAMA_BASE_URL=http://localhost:11439
```

## Test Evaluation

### Compare Qwen Base vs Finetuned

To compare the Qwen 0.5 base model against the finetuned model:

**1. Start both servers:**

Terminal 1 (Finetuned):
```bash
python text2sql/scripts/qwen_server.py
```

Terminal 2 (Base):
```bash
python text2sql/scripts/qwen_base_server.py
```

**2. Run the comparison:**

```bash
python text2sql/scripts/test_qwen_baseline_comparison.py
```

This will:
- Evaluate both models on 35 test cases
- Test all 4 prompt techniques (Few-Shot, CoT, LtM, EG)
- Compare execution accuracy, latency, and success rates
- Save results to `text2sql/scripts/data/results/`

### Other Evaluation Scripts

**Compare RAG Impact:**
```bash
python text2sql/scripts/compare_rag.py
```

**Generate Comprehensive Evaluation:**
```bash
python text2sql/scripts/generate_100_evaluation.py
```

## Pushing Code

To push the code with the finetuned model for others to test:

```bash
# Add new files
git add text2sql/scripts/qwen*.py
git add text2sql/scripts/test_qwen*.py
git add text2sql/model_client.py
git add .gitignore
git add qwen0p5b-spider-lora/

# Commit
git commit -m "Add Qwen 0.5 baseline testing and finetuned model"

# Push
git push
```

**Note:** The finetuned model adapter (~16.5 MB) is included in the repository. Checkpoints are excluded via `.gitignore`.

## Evaluation Scripts

### Compare RAG Impact

```bash
python text2sql/scripts/compare_rag.py
```

Compares performance with and without RAG schema retrieval.

### Generate Comprehensive Evaluation

```bash
python text2sql/scripts/generate_100_evaluation.py
```

Runs 100 test cases and generates comparison reports.

## Performance Notes

- **Arctic Fine-tuned**: ~2-5 seconds per query (depends on GPU)
- **GPT-4o-mini**: ~1-3 seconds per query (depends on API latency)
- **Database queries**: <100ms for most queries

## License

[Add your license here]

## Citation

If you use this project, please cite:

- **Arctic-Text2SQL**: Snowflake's Arctic-Text2SQL-R1-7B model
- **Spider Dataset**: Used for fine-tuning (if applicable)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review error messages in terminal/console
3. Verify all prerequisites are installed
4. Check `.env` configuration

## Quick Start Checklist

- [ ] Install Python 3.10+
- [ ] Install dependencies: `pip install -r text2sql/requirements.txt`
- [ ] Set up `.env` file with API keys
- [ ] Set up database: `python text2sql/scripts/setup_database.py`
- [ ] For Arctic model: Install CUDA dependencies
- [ ] For Arctic model: Start server: `python text2sql/scripts/finetuned_arctic_server.py`
- [ ] Start Streamlit: `streamlit run text2sql/app.py`
- [ ] Test with a query!

---

**Happy SQL Generation! 🚀**

