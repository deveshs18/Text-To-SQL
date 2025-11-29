"""
Server for fine-tuned Arctic model with LoRA adapters.
Serves the fine-tuned model via Ollama-compatible API.
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

app = Flask(__name__)
model_lock = Lock()
model = None
tokenizer = None

BASE_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"
LORA_MODEL = "arctic_lora_model"

def load_finetuned_model():
    """Load base model with LoRA adapters."""
    global model, tokenizer
    
    print("="*80)
    print("LOADING FINE-TUNED ARCTIC MODEL")
    print("="*80)
    
    if not os.path.exists(LORA_MODEL):
        raise FileNotFoundError(f"LoRA adapters not found: {LORA_MODEL}")
    
    print(f"[1/3] Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Base model loaded!")
    
    print(f"\n[2/3] Loading LoRA adapters from: {LORA_MODEL}")
    model = PeftModel.from_pretrained(model, LORA_MODEL)
    print("✅ LoRA adapters loaded!")
    
    # Read training summary if available
    summary_path = os.path.join(LORA_MODEL, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"\n📊 Training Summary:")
        print(f"   Dataset size: {summary.get('dataset_size', 'N/A')} examples")
        print(f"   Epochs: {summary.get('epochs', 'N/A')}")
        print(f"   Trainable params: {summary.get('trainable_percentage', 'N/A')}")
    
    print("\n[3/3] Model ready for inference!")
    return model, tokenizer

@app.route('/api/generate', methods=['POST'])
def generate():
    """Ollama-compatible generate endpoint."""
    global model, tokenizer
    
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('options', {}).get('temperature', 0.1)
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    start_time = time.time()
    
    try:
        # Use prompt as-is - model was trained on Alpaca format!
        # Don't strip the format - the fine-tuned model expects it
        processed_prompt = prompt
        
        # If prompt doesn't have Alpaca format, add it (for compatibility)
        if "### Instruction:" not in prompt and "### Input:" not in prompt:
            # Simple format - convert to Alpaca format that model was trained on
            if "SCHEMA:" in prompt and "Q:" in prompt:
                # Extract schema and question
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
            inputs = tokenizer(processed_prompt, return_tensors="pt", padding=False, truncation=False).to(model.device)
            
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
        input_length = inputs['input_ids'].shape[1]
        generated_token_ids = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        
        # The model should generate SQL directly in the Response section
        # Extract from "### Response:" if present, otherwise use the text as-is
        if "### Response:" in response_text:
            response_text = response_text.split("### Response:")[-1].strip()
        elif "Response:" in response_text:
            response_text = response_text.split("Response:")[-1].strip()
        
        # Clean up response - remove any remaining markers
        response_text = re.sub(r'^###\s*Response:\s*', '', response_text, flags=re.IGNORECASE).strip()
        response_text = re.sub(r'^SQL:\s*', '', response_text, flags=re.IGNORECASE).strip()
        
        # Remove explanation prefixes (model might add these)
        explanation_patterns = [
            r'^To translate.*?into.*?SQL.*?:',
            r'^The question asks.*?:',
            r'^We need to.*?:',
            r'^Given.*?:',
        ]
        for pattern in explanation_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Stop at section markers
        for marker in ["\n###", "\nQ:", "\nSQL:", "\nassistant", "\n\n\n"]:
            if marker in response_text:
                response_text = response_text.split(marker)[0].strip()
        
        # Try to extract SQL if buried in text
        sql_match = re.search(r'(SELECT|WITH)\s+.*?(?=\n\n|\n###|\nQ:|\nSQL:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            response_text = sql_match.group(0).strip()
        
        # Fix common issues
        if response_text.upper().startswith('ELECT'):
            response_text = 'S' + response_text
            print("⚠️  Fixed missing 'S' in SELECT")
        
        # Extract SQL line by line
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
                   line.startswith(('FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'ON', 'AND', 'OR', 'AS', '(', ')', ',', ';')):
                    sql_lines.append(line)
                elif not line[0].isalpha() or line[0].islower():
                    sql_lines.append(line)
                else:
                    break
        
        if sql_lines:
            response_text = ' '.join(sql_lines).strip()
        
        # Ensure semicolon
        if response_text and not response_text.endswith(';'):
            sql_upper = response_text.upper()
            if not any(sql_upper.endswith(ending) for ending in ['LIMIT', 'DESC', 'ASC', ')']):
                if sql_upper.startswith(('SELECT', 'WITH')):
                    response_text = response_text.rstrip() + ';'
        
        latency = time.time() - start_time
        
        print("\n" + "="*40)
        print("DEBUG: PROMPT RECEIVED (last 200 chars)")
        print("-" * 20)
        print(prompt[-200:])
        print("-" * 20)
        print("DEBUG: RAW GENERATION (full)")
        print("-" * 20)
        print(response_text)
        print("-" * 20)
        print(f"SQL Length: {len(response_text)} chars")
        print(f"Ends with ';': {response_text.endswith(';')}")
        print("="*40 + "\n")
        
        return jsonify({
            'response': response_text,
            'done': True,
            'total_duration': int(latency * 1e9),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags', methods=['GET'])
def tags():
    """Ollama-compatible tags endpoint."""
    return jsonify({
        'models': [{
            'name': 'arctic-finetuned',
            'modified_at': '2025-01-01T00:00:00Z',
            'size': 0,
            'digest': 'arctic-finetuned'
        }]
    })

if __name__ == '__main__':
    print("="*80)
    print("FINE-TUNED ARCTIC MODEL INFERENCE SERVER")
    print("="*80)
    print("This server exposes your fine-tuned model via an Ollama-compatible API.")
    print("Your Streamlit app can connect to this instead of Ollama.")
    print()
    
    try:
        model, tokenizer = load_finetuned_model()
        
        print()
        print("="*80)
        print("SERVER STARTING")
        print("="*80)
        print("URL: http://localhost:11437")
        print("To use in your app, set in .env:")
        print("  MODEL_NAME=ollama/arctic-finetuned")
        print("  OLLAMA_BASE_URL=http://localhost:11437")
        print("="*80)
        print()
        
        app.run(host='0.0.0.0', port=11437, debug=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

