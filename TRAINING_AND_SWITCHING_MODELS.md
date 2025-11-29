# Guide: Training Your Own Model and Switching Between Models

This guide explains how to:
1. Train your own fine-tuned model
2. Create a server for your new model
3. Switch between your model and GPT-4o-mini in Streamlit
4. Compare results

## Step 1: Train Your Own Model

### 1.1 Prepare Your Dataset

Your dataset should be in Alpaca format (JSON file). Example structure:

```json
[
  {
    "instruction": "You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question.",
    "input": "SCHEMA: users(id INT, name TEXT, email TEXT)\nQ: How many users are there?\nSQL:",
    "output": "SELECT COUNT(*) FROM users;"
  },
  ...
]
```

**File location:** `ft_spider/data/your_dataset.json`

### 1.2 Configure Training Script

Edit `text2sql/scripts/finetune_arctic.py`:

```python
# Configuration (around line 26-29)
BASE_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"  # Keep this or change to your base model
OUTPUT_DIR = "your_model_lora"  # CHANGE THIS: Name for your model adapters
DATASET_PATH = "ft_spider/data/your_dataset.json"  # CHANGE THIS: Your dataset path
TRAINING_SUBSET_SIZE = None  # Set to None for full dataset, or number for subset
```

**Key parameters to adjust:**
- `OUTPUT_DIR`: Directory where your LoRA adapters will be saved
- `DATASET_PATH`: Path to your training dataset
- `TRAINING_SUBSET_SIZE`: Number of examples (None = all)
- `num_train_epochs`: In `TrainingArguments` (line ~138) - default is 1
- `learning_rate`: In `TrainingArguments` (line ~142) - default is 1e-4

### 1.3 Run Training

```bash
python text2sql/scripts/finetune_arctic.py
```

**What happens:**
- Downloads base model from Hugging Face (first time only)
- Loads your dataset
- Trains LoRA adapters
- Saves adapters to `your_model_lora/` directory

**Training time:**
- 500 examples, 1 epoch: ~1-2 hours (RTX 4060)
- Full dataset (8,659 examples), 1 epoch: ~12-15 hours (RTX 4060)

## Step 2: Create Server for Your New Model

### 2.1 Copy and Modify Server Script

**Option A: Create a new server file (Recommended)**

Create `text2sql/scripts/your_model_server.py` by copying `finetuned_arctic_server.py`:

```bash
# Copy the existing server
cp text2sql/scripts/finetuned_arctic_server.py text2sql/scripts/your_model_server.py
```

**Option B: Modify existing server**

Edit `text2sql/scripts/finetuned_arctic_server.py`:

```python
# Change these lines (around line 20-21):
BASE_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"  # Your base model
LORA_MODEL = "your_model_lora"  # CHANGE THIS: Your model directory name
```

And change the port (around line 260):

```python
app.run(host='0.0.0.0', port=11438, debug=False)  # CHANGE PORT: Use different port (11438, 11439, etc.)
```

### 2.2 Update Server Port

**Important:** Use a different port for each model to avoid conflicts.

In your server file, change the port:

```python
# In finetuned_arctic_server.py or your_model_server.py
app.run(host='0.0.0.0', port=11438, debug=False)  # Use unique port
```

**Port assignments:**
- Arctic fine-tuned (existing): Port 11437
- Your new model: Port 11438 (or any available port)
- GPT-4o-mini: No server needed (uses OpenAI API)

### 2.3 Test Your Server

```bash
python text2sql/scripts/your_model_server.py
```

Expected output:
```
================================================================================
LOADING FINE-TUNED ARCTIC MODEL
================================================================================
[1/3] Loading base model: Snowflake/Arctic-Text2SQL-R1-7B
✅ Base model loaded!

[2/3] Loading LoRA adapters from: your_model_lora
✅ LoRA adapters loaded!

[3/3] Model ready for inference!
================================================================================
SERVER STARTING
================================================================================
URL: http://localhost:11438
```

## Step 3: Configure Streamlit App to Switch Models

### 3.1 Update Model Client (Optional - for automatic detection)

Edit `text2sql/model_client.py`:

**No changes needed** - The model client already supports switching via `.env` file.

### 3.2 Update Prompts (Optional - if your model needs different format)

Edit `text2sql/prompts.py`:

**Only needed if your model uses a different prompt format.**

In `format_for_model()` function (around line 210), add detection for your model:

```python
# Check if using your custom model (port 11438)
is_your_model = (
    "your_model" in model_name.lower() or
    os.getenv("OLLAMA_BASE_URL", "").endswith("11438")  # Your server port
)

if is_your_model:
    # Use format that matches your training data
    return f"""Below is an instruction that describes a task...

### Instruction:
{instruction}

### Input:
{content}
SQL:

### Response:
"""
```

**Note:** If your model was trained with the same format as Arctic, no changes needed.

### 3.3 Switch Models via .env File

**Method 1: Edit .env file**

Create or edit `.env` in project root:

**For your new model:**
```bash
MODEL_NAME=ollama/your-model-name
OLLAMA_BASE_URL=http://localhost:11438
```

**For GPT-4o-mini:**
```bash
MODEL_NAME=openai/gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
```

**Method 2: Use Streamlit UI**

The Streamlit app has a sidebar where you can change the model name directly:

1. Start Streamlit: `streamlit run text2sql/app.py`
2. In the sidebar, find "Model Name" field
3. Change it to:
   - `ollama/your-model-name` (for your model)
   - `openai/gpt-4o-mini` (for GPT-4o-mini)
4. Also update "OLLAMA_BASE_URL" if using your model (e.g., `http://localhost:11438`)

## Step 4: Running and Testing

### 4.1 Start Your Model Server

**Terminal 1:**
```bash
python text2sql/scripts/your_model_server.py
```

Keep this running.

### 4.2 Start Streamlit App

**Terminal 2:**
```bash
streamlit run text2sql/app.py
```

### 4.3 Test Your Model

1. Open browser: `http://localhost:8501`
2. In sidebar, set:
   - Model Name: `ollama/your-model-name`
   - OLLAMA_BASE_URL: `http://localhost:11438` (if not in .env)
3. Enter a query
4. Click "Generate SQL"
5. View results

### 4.4 Switch to GPT-4o-mini

1. In Streamlit sidebar, change:
   - Model Name: `openai/gpt-4o-mini`
2. Enter the same query
3. Click "Generate SQL"
4. Compare results

### 4.5 Compare Results

Use the "Saved Results" tab in Streamlit to:
- View query history
- Compare SQL from different models
- Check metrics (Exact Match, Execution Accuracy)

## Quick Reference: File Changes Summary

### Files to Modify for New Model:

1. **`text2sql/scripts/finetune_arctic.py`**
   - Line ~28: `OUTPUT_DIR = "your_model_lora"`
   - Line ~29: `DATASET_PATH = "ft_spider/data/your_dataset.json"`
   - Line ~138: `num_train_epochs` (optional)
   - Line ~142: `learning_rate` (optional)

2. **`text2sql/scripts/your_model_server.py`** (new file or modify existing)
   - Line ~20: `BASE_MODEL` (if different base model)
   - Line ~21: `LORA_MODEL = "your_model_lora"`
   - Line ~260: `port=11438` (unique port)

3. **`.env` file** (create or edit)
   - `MODEL_NAME=ollama/your-model-name`
   - `OLLAMA_BASE_URL=http://localhost:11438`

4. **`text2sql/prompts.py`** (optional - only if different prompt format)
   - Add detection for your model port
   - Add format function if needed

### Files That DON'T Need Changes:

- `text2sql/app.py` - Works with any model via .env
- `text2sql/model_client.py` - Already supports switching
- `text2sql/db.py` - Database utilities (unchanged)
- `text2sql/schema_retriever.py` - Schema retrieval (unchanged)

## Example Workflow

### Training Your Model:

```bash
# 1. Prepare dataset
# Place your_dataset.json in ft_spider/data/

# 2. Edit finetune_arctic.py
# Change OUTPUT_DIR and DATASET_PATH

# 3. Train
python text2sql/scripts/finetune_arctic.py

# Wait for training to complete...
# Adapters saved to: your_model_lora/
```

### Setting Up Server:

```bash
# 1. Copy server script
cp text2sql/scripts/finetuned_arctic_server.py text2sql/scripts/your_model_server.py

# 2. Edit your_model_server.py
# Change LORA_MODEL and port

# 3. Test server
python text2sql/scripts/your_model_server.py
```

### Testing in Streamlit:

```bash
# Terminal 1: Start your model server
python text2sql/scripts/your_model_server.py

# Terminal 2: Start Streamlit
streamlit run text2sql/app.py

# In browser:
# 1. Set MODEL_NAME=ollama/your-model-name
# 2. Set OLLAMA_BASE_URL=http://localhost:11438
# 3. Test queries

# To switch to GPT-4o-mini:
# 1. Change MODEL_NAME=openai/gpt-4o-mini
# 2. Test same queries
# 3. Compare results
```

## Troubleshooting

### Model Server Won't Start

- **Check port is available:** Use `netstat -an | findstr 11438` (Windows) or `lsof -i :11438` (Linux/Mac)
- **Check LoRA adapters exist:** `ls your_model_lora/` should show `adapter_model.safetensors`
- **Check CUDA:** `python -c "import torch; print(torch.cuda.is_available())"`

### Streamlit Can't Connect to Model

- **Verify server is running:** Check terminal for "SERVER STARTING" message
- **Check port matches:** Server port must match `OLLAMA_BASE_URL` port
- **Check model name:** Must start with `ollama/` for local servers

### Model Generates Wrong SQL

- **Check prompt format:** Ensure `prompts.py` format matches your training data
- **Check training quality:** Model might need more training or better data
- **Compare with base model:** Test if base Arctic model works better

## Tips

1. **Use descriptive model names:** `ollama/my-custom-text2sql-v1`
2. **Document your training:** Note dataset size, epochs, learning rate
3. **Test incrementally:** Start with small dataset, then scale up
4. **Save checkpoints:** Training script saves checkpoints automatically
5. **Compare systematically:** Test same queries on both models

## Summary

**To train and use your own model:**
1. Modify `finetune_arctic.py` → Train → Get `your_model_lora/`
2. Create/modify server script → Change port → Start server
3. Update `.env` or Streamlit UI → Switch model name
4. Test and compare!

**To switch between models:**
- Just change `MODEL_NAME` in `.env` or Streamlit sidebar
- No code changes needed (if using same prompt format)

