"""
Server for fine-tuned Arctic model with LoRA adapters.
Serves the fine-tuned model via:
  - Ollama-compatible API: /api/generate
  - OpenAI-compatible API: /v1/chat/completions
"""

import os
import re
import time
import json
import torch
from flask import Flask, request, jsonify
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Optional small speed boost on modern GPUs
torch.set_float32_matmul_precision("high")

app = Flask(__name__)
model_lock = Lock()
model = None
tokenizer = None

BASE_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"
LORA_MODEL = "arctic_lora_model"


# ==============================
#  MODEL LOADING
# ==============================

def load_finetuned_model():
    """Load base model with LoRA adapters."""
    global model, tokenizer

    print("=" * 80)
    print("LOADING FINE-TUNED ARCTIC MODEL")
    print("=" * 80)

    if not os.path.exists(LORA_MODEL):
        raise FileNotFoundError(f"LoRA adapters not found: {LORA_MODEL}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference with this model.")

    print(f"[1/3] Loading base model: {BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_local = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tok_local = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tok_local.pad_token = tok_local.eos_token
    tok_local.padding_side = "right"

    print("✅ Base model loaded!")

    print(f"\n[2/3] Loading LoRA adapters from: {LORA_MODEL}")
    model_local = PeftModel.from_pretrained(model_local, LORA_MODEL)
    print("✅ LoRA adapters loaded!")

    # Read training summary if available
    summary_path = os.path.join(LORA_MODEL, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print("\n📊 Training Summary:")
        print(f"   Dataset size: {summary.get('dataset_size', 'N/A')} examples")
        print(f"   Epochs: {summary.get('epochs', 'N/A')}")
        print(f"   Trainable params: {summary.get('trainable_percentage', 'N/A')}")

    print("\n[3/3] Model ready for inference!")

    return model_local, tok_local


# ==============================
#  CORE GENERATION LOGIC
# ==============================

def generate_sql_from_prompt(prompt: str, temperature: float = 0.1) -> str:
    """
    Core generation function used by both:
      - /api/generate (Ollama-style)
      - /v1/chat/completions (OpenAI-style)
    It expects the prompt to already be in the correct format
    (your training Alpaca-style formatter will handle that).
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # If prompt is not in Alpaca format but contains SCHEMA & Q,
    # convert into the training format.
    processed_prompt = prompt

    if "### Instruction:" not in prompt and "### Input:" not in prompt:
        if "SCHEMA:" in prompt and "Q:" in prompt:
            schema_match = re.search(r'SCHEMA:\s*(.*?)(?=\nQ:|$)', prompt, re.DOTALL)
            question_match = re.search(r'Q:\s*(.*?)(?=\nSQL:|$)', prompt, re.DOTALL)
            if schema_match and question_match:
                schema = schema_match.group(1).strip()
                question = question_match.group(1).strip()
                processed_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a powerful text-to-SQL model. Your job is to generate valid SQL queries for the given schema and question.

### Input:
SCHEMA: {schema}
Q: {question}
SQL:

### Response:
"""
                print("🔧 Converted simple format to Alpaca format (training format)")

    with model_lock:
        inputs = tokenizer(
            processed_prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

    # Extract generated text (only the new tokens after the prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_token_ids = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    # If model echoed training structure, strip to just SQL
    if "### Response:" in response_text:
        response_text = response_text.split("### Response:")[-1].strip()
    elif "Response:" in response_text:
        response_text = response_text.split("Response:")[-1].strip()

    # Clean markers
    response_text = re.sub(r'^###\s*Response:\s*', '', response_text, flags=re.IGNORECASE).strip()
    response_text = re.sub(r'^SQL:\s*', '', response_text, flags=re.IGNORECASE).strip()

    # Remove typical explanation intros if present
    explanation_patterns = [
        r'^To translate.*?into.*?SQL.*?:',
        r'^The question asks.*?:',
        r'^We need to.*?:',
        r'^Given.*?:',
    ]
    for pattern in explanation_patterns:
        response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()

    # Truncate at any obvious section markers
    for marker in ["\n###", "\nQ:", "\nSQL:", "\nassistant", "\n\n\n"]:
        if marker in response_text:
            response_text = response_text.split(marker)[0].strip()

    # Try to pull out SQL if it's inside text
    sql_match = re.search(
        r'(SELECT|WITH)\s+.*?(?=\n\n|\n###|\nQ:|\nSQL:|$)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if sql_match:
        response_text = sql_match.group(0).strip()

    # Fix "ELECT" → "SELECT"
    if response_text.upper().startswith("ELECT"):
        response_text = "S" + response_text
        print("⚠️  Fixed missing 'S' in SELECT")

    # Line-wise extraction to avoid stray text
    sql_lines = []
    found_sql_start = False

    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        line_upper = line.upper()
        if line_upper.startswith('SELECT') or line_upper.startswith('WITH'):
            found_sql_start = True
            sql_lines.append(line)
        elif line_upper.startswith('ELECT'):
            found_sql_start = True
            sql_lines.append('S' + line)
        elif found_sql_start:
            if any(marker in line for marker in ['###', 'Q:', 'SQL:', 'assistant', 'To translate']):
                break
            if re.match(r'^[A-Z_][A-Z0-9_]*\s*[,=<>!]', line, re.IGNORECASE) or \
               line.startswith((
                   'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'JOIN',
                   'INNER', 'LEFT', 'RIGHT', 'ON', 'AND', 'OR', 'AS',
                   '(', ')', ',', ';'
               )):
                sql_lines.append(line)
            elif not line[0].isalpha() or line[0].islower():
                sql_lines.append(line)
            else:
                break

    if sql_lines:
        response_text = ' '.join(sql_lines).strip()

    # Ensure trailing semicolon if it's a SELECT/WITH
    if response_text and not response_text.endswith(';'):
        sql_upper = response_text.upper()
        if sql_upper.startswith(('SELECT', 'WITH')):
            if not any(sql_upper.endswith(ending) for ending in ['LIMIT', 'DESC', 'ASC', ')']):
                response_text = response_text.rstrip() + ';'

    return response_text


# ==============================
#  OLLAMA-COMPATIBLE ENDPOINT
# ==============================

@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama-compatible generate endpoint: expects {'prompt': '...'}"""
    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = data.get("options", {}).get("temperature", 0.1)

    start_time = time.time()

    try:
        sql = generate_sql_from_prompt(prompt, temperature=float(temperature))
        latency = time.time() - start_time

        print("\n" + "=" * 40)
        print("DEBUG /api/generate: SQL")
        print(sql)
        print("=" * 40 + "\n")

        return jsonify({
            "response": sql,
            "done": True,
            "total_duration": int(latency * 1e9),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
#  OPENAI-COMPATIBLE ENDPOINT
# ==============================

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.
    Expects:
    {
      "model": "arctic-finetuned",
      "messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
      "temperature": 0.1
    }
    """
    data = request.json or {}
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", 0.1))
    model_name = data.get("model", "arctic-finetuned")

    # Build a single prompt from messages (simple concat).
    # Your upstream code already formats proper Alpaca-style prompts,
    # so usually you'll just send that as one user message.
    system_parts = []
    user_parts = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)

    # Simple strategy: prepend system to user text
    if system_parts:
        prompt = "\n".join(system_parts) + "\n\n" + "\n".join(user_parts)
    else:
        prompt = "\n".join(user_parts)

    start_time = time.time()

    try:
        sql = generate_sql_from_prompt(prompt, temperature=temperature)
        latency = time.time() - start_time

        # Minimal OpenAI-style response
        now = int(time.time())
        response = {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": sql,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                # If you need real token counts, we can add them later.
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency_ms": int(latency * 1000),
        }

        print("\n" + "=" * 40)
        print("DEBUG /v1/chat/completions: SQL")
        print(sql)
        print("=" * 40 + "\n")

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
#  OLLAMA TAGS ENDPOINT
# ==============================

@app.route("/api/tags", methods=["GET"])
def tags():
    """Ollama-compatible tags endpoint."""
    return jsonify({
        "models": [{
            "name": "arctic-finetuned",
            "modified_at": "2025-01-01T00:00:00Z",
            "size": 0,
            "digest": "arctic-finetuned"
        }]
    })


# ==============================
#  MAIN
# ==============================

if __name__ == "__main__":
    print("=" * 80)
    print("FINE-TUNED ARCTIC MODEL INFERENCE SERVER")
    print("=" * 80)
    print("This server exposes your fine-tuned model via:")
    print("  - Ollama-compatible API:   POST http://localhost:11437/api/generate")
    print("  - OpenAI-compatible API:   POST http://localhost:11437/v1/chat/completions")
    print()

    try:
        model, tokenizer = load_finetuned_model()

        print()
        print("=" * 80)
        print("SERVER STARTING")
        print("=" * 80)
        print("URL: http://localhost:11437")
        print()
        print("For your Streamlit / RAG app, you can use:")
        print("  OLLAMA_BASE_URL=http://localhost:11437")
        print("  MODEL_NAME=arctic-finetuned  (for /v1/chat/completions)")
        print("=" * 80)
        print()

        app.run(host="0.0.0.0", port=11437, debug=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
