"""
Hardened inference server for Qwen2.5 7B + LoRA text-to-SQL model
Prevents hangs, self-join explosion queries, and invalid SQL output.
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

torch.set_float32_matmul_precision("high")

app = Flask(__name__)

model_lock = Lock()
model = None
tokenizer = None

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL = "qwen2_5_7b_text2sql_lora"


# ==========================================================
#  MODEL LOADING
# ==========================================================

def load_finetuned_model():
    global model, tokenizer

    print("=" * 80)
    print("LOADING QWEN MODEL")
    print("=" * 80)

    use_lora = os.path.isdir(LORA_MODEL)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_local = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tok_local = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )

    if tok_local.pad_token is None:
        tok_local.pad_token = tok_local.eos_token
    tok_local.padding_side = "right"

    if use_lora:
        print(f"Loading LoRA: {LORA_MODEL}")
        model_local = PeftModel.from_pretrained(model_local, LORA_MODEL)

    print("Model ready.")
    return model_local, tok_local


# ==========================================================
#  SQL EXTRACTION SAFETY ENGINE
# ==========================================================

def extract_sql_only(text):
    """
    Extracts SQL only, removes chat, scaffolding, and hallucination text.
    """
    text = text.strip()

    # Strip Alpaca formatting noise
    for pat in ["### Response:", "Response:", "SQL:", "Query:"]:
        text = re.sub(rf"^{pat}\s*", "", text, flags=re.IGNORECASE)

    # Extract SELECT/WITH query
    match = re.search(r"(SELECT|WITH)[\s\S]*?(;|$)", text, re.IGNORECASE)
    sql = match.group(0).strip() if match else text

    # Fix ELECT
    if sql.upper().startswith("ELECT"):
        sql = "S" + sql

    # Remove sentences before SQL
    if "\n" in sql:
        first_line = sql.split("\n")[0]
        if not first_line.strip().upper().startswith(("SELECT", "WITH")):
            sql = re.sub(r".*?(SELECT|WITH)", r"\1", sql, flags=re.IGNORECASE | re.DOTALL)

    # Remove trailing commentary
    for marker in ["###", "assistant", "Instruction", "Rewrite", "context"]:
        if marker.lower() in sql.lower():
            sql = sql.split(marker)[0].strip()

    # Ensure semicolon
    if not sql.endswith(";"):
        if sql.upper().startswith(("SELECT", "WITH")):
            sql += ";"

    return sql.strip()


def query_is_dangerous(sql):
    """
    Detects infinite-loop / explosion queries such as:
    - self joins
    - double table scans
    - CROSS PRODUCT attempts
    """
    upper = sql.upper()
    if sql.count("ADULT_INCOME") > 1:
        return True
    if "JOIN ADULT_INCOME" in upper:
        return True
    if " CROSS " in upper and "JOIN" not in upper:
        return True
    if upper.count(" FROM ") > 1:
        return True
    return False


# ==========================================================
#  GENERATION LOGIC
# ==========================================================

def generate_sql_from_prompt(prompt: str, temperature: float = 0.1) -> str:
    global model, tokenizer

    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # Convert simple SCHEMA/Q format to Alpaca format
    processed_prompt = prompt
    if "### Instruction" not in prompt and "SCHEMA:" in prompt and "Q:" in prompt:
        schema_match = re.search(r"SCHEMA:\s*(.*?)(?=\nQ:|$)", prompt, re.DOTALL)
        q_match = re.search(r"Q:\s*(.*?)(?=\n|$)", prompt, re.DOTALL)
        if schema_match and q_match:
            processed_prompt = f"""Below is an instruction.

### Instruction:
Generate executable SQL only.

### Input:
SCHEMA: {schema_match.group(1).strip()}
Q: {q_match.group(1).strip()}
SQL:

### Response:
"""
            print("🔧 Rewrapped prompt -> Alpaca")

    # Inference
    with model_lock:
        inputs = tokenizer(processed_prompt,
                           return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=280,
                temperature=max(0.05, temperature),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.12,
            )

    # Decode
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    sql = extract_sql_only(raw_text)

    # Reject and force regen if dangerous
    if query_is_dangerous(sql) or len(sql) < 8 or not sql.upper().startswith(("SELECT", "WITH")):
        print("⚠️ SQL rejected – regenerating")
        return ""

    return sql


# ==========================================================
#  REGEN LOGIC (prevents hangs)
# ==========================================================

def safe_generate(prompt, temperature):
    sql = generate_sql_from_prompt(prompt, temperature)
    retries = 0
    while (not sql or len(sql) < 6) and retries < 3:
        print("🔁 Regenerating...")
        sql = generate_sql_from_prompt(prompt, temperature)
        retries += 1
    if not sql:
        sql = "SELECT 1;"  # worst-case fallback
    return sql


# ==========================================================
#  API ENDPOINTS
# ==========================================================

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = float(data.get("options", {}).get("temperature", 0.1))
    start = time.time()

    sql = safe_generate(prompt, temperature)
    return jsonify({
        "response": sql,
        "done": True,
        "total_duration": int((time.time() - start) * 1e9),
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json or {}
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", 0.1))
    model_name = data.get("model", "arctic-finetuned")

    system_parts, user_parts = [], []
    for m in messages:
        (system_parts if m.get("role") == "system" else user_parts).append(m.get("content", ""))

    prompt = ("\n".join(system_parts) + "\n\n" if system_parts else "") + "\n".join(user_parts)

    start = time.time()
    sql = safe_generate(prompt, temperature)

    now = int(time.time())
    return jsonify({
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": sql},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": int((time.time() - start) * 1000),
    })


@app.route("/api/tags", methods=["GET"])
def tags():
    return jsonify({
        "models": [{
            "name": "arctic-finetuned",
            "modified_at": "2025-01-01T00:00:00Z",
            "size": 0,
            "digest": "arctic-finetuned"
        }]
    })


# ==========================================================
#  MAIN
# ==========================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TEXT-TO-SQL INFERENCE SERVER")
    print("=" * 80)

    try:
        model, tokenizer = load_finetuned_model()
        app.run(host="0.0.0.0", port=11437, debug=False)
    except Exception as e:
        print("❌ Model load error:", e)
        import traceback
        traceback.print_exc()
