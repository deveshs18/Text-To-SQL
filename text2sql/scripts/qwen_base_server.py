"""
Server for base Qwen2.5-Coder-0.5B model (no fine-tuning).
Serves the base model directly from Hugging Face via Ollama-compatible API.
"""
import os
import re
import time
import torch
from flask import Flask, request, jsonify
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = Flask(__name__)
model_lock = Lock()
model = None
tokenizer = None

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B"

def load_base_model():
    """Load base model directly from Hugging Face (no LoRA adapters)."""
    global model, tokenizer
    
    print("="*80)
    print("LOADING BASE QWEN MODEL")
    print("="*80)
    print("This will download the model from Hugging Face if not already cached.")
    print()
    
    print(f"[1/2] Loading base model: {BASE_MODEL}")
    print("This may take a few minutes on first run (downloading ~1GB)...")
    
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
        torch_dtype=torch.bfloat16,
    )
    
    print("\n[2/2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Optimize model for inference
    model.eval()
    
    # Try to compile model for faster inference (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            print("Optimizing model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compiled for faster inference!")
    except Exception as e:
        print(f"⚠️  Could not compile model: {e} (continuing without compilation)")
    
    print("✅ Base model loaded!")
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
        # Qwen model expects format: SCHEMA: ... Q: ... SQL: ...
        processed_prompt = prompt
        
        # If prompt has Alpaca format, extract the actual content
        if "### Instruction:" in prompt and "### Input:" in prompt:
            # Extract content from Input section
            input_match = re.search(r'### Input:\s*(.*?)(?=\n### Response:|$)', prompt, re.DOTALL)
            if input_match:
                processed_prompt = input_match.group(1).strip()
                # Ensure it ends with SQL: if not already
                if not processed_prompt.rstrip().endswith('SQL:'):
                    processed_prompt = processed_prompt.rstrip() + '\nSQL:'
        elif "SCHEMA:" not in prompt or "Q:" not in prompt:
            # Try to convert simple format
            if "SCHEMA:" in prompt and "Q:" in prompt:
                # Already in correct format, just ensure SQL: at end
                if not prompt.rstrip().endswith('SQL:'):
                    processed_prompt = prompt.rstrip() + '\nSQL:'
        
        with model_lock:
            inputs = tokenizer(processed_prompt, return_tensors="pt", padding=False, truncation=False).to(model.device)
            
            # Use inference_mode for faster inference
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # SQL queries are usually short
                    temperature=temperature if temperature > 0 else 0.0,
                    do_sample=False,  # Greedy decoding for speed
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    use_cache=True,
                    num_beams=1,
                )
        
        # Extract generated text (only the new tokens after the prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_token_ids = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        
        # Clean up response - remove any markers
        response_text = re.sub(r'^SQL:\s*', '', response_text, flags=re.IGNORECASE).strip()
        response_text = re.sub(r'^###\s*Response:\s*', '', response_text, flags=re.IGNORECASE).strip()
        
        # Remove explanation prefixes
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
        
        # Extract SQL if buried in text
        sql_match = re.search(r'(SELECT|WITH)\s+.*?(?=\n\n|\n###|\nQ:|\nSQL:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            response_text = sql_match.group(0).strip()
        
        # Fix common issues
        if response_text.upper().startswith('ELECT'):
            response_text = 'S' + response_text
        
        # Extract SQL line by line
        sql_lines = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if line.startswith('--') or line.startswith('#'):
                continue
            # Stop at certain markers
            if any(marker in line.upper() for marker in ['EXPLANATION', 'NOTE:', 'HERE', 'ANSWER:']):
                break
            sql_lines.append(line)
        
        if sql_lines:
            response_text = ' '.join(sql_lines).strip()
        
        # Remove trailing backticks and markdown formatting
        response_text = re.sub(r'[`]+;?\s*$', '', response_text).strip()
        response_text = re.sub(r'```\s*;?\s*$', '', response_text).strip()
        response_text = re.sub(r'`+\s*$', '', response_text).strip()
        
        # Ensure semicolon (after cleaning backticks)
        if response_text and not response_text.endswith(';'):
            sql_upper = response_text.upper()
            if not any(sql_upper.endswith(ending) for ending in ['LIMIT', 'DESC', 'ASC', ')']):
                if sql_upper.startswith(('SELECT', 'WITH')):
                    response_text = response_text.rstrip() + ';'
        
        latency = time.time() - start_time
        
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
            'name': 'qwen-0.5b-base',
            'modified_at': '2025-01-01T00:00:00Z',
            'size': 0,
            'digest': 'qwen-0.5b-base'
        }]
    })

if __name__ == '__main__':
    print("="*80)
    print("BASE QWEN MODEL INFERENCE SERVER")
    print("="*80)
    print("This server exposes the base Qwen2.5-Coder-0.5B model via an Ollama-compatible API.")
    print("No fine-tuning - using the model directly from Hugging Face.")
    print()
    
    # Load model
    try:
        model, tokenizer = load_base_model()
        if model is None or tokenizer is None:
            print("Failed to load model. Exiting.")
            exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*80)
    print("SERVER STARTING")
    print("="*80)
    print("URL: http://localhost:11439")
    print("Model name: ollama/qwen-0.5b-base")
    print("="*80)
    print()
    
    # Suppress Flask startup banner to avoid Windows console errors
    import sys
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    try:
        app.run(host='0.0.0.0', port=11439, debug=False)
    except OSError as e:
        if e.winerror == 6:  # Windows error 6: Invalid handle
            print("Server started (Windows console error can be ignored)")
        else:
            raise

