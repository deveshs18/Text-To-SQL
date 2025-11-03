"""Model client: Ollama and OpenAI backends for SQL generation."""
import os
import re
import time
import requests
from typing import Optional, Tuple, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_sql_from_output(text: str) -> str:
    """
    Extract SQL from model output.
    Looks for SQL: marker, ```sql blocks, or returns trimmed output.
    Returns extracted SQL, or raises ValueError if SQL doesn't start with SELECT/WITH.
    """
    text = text.strip()
    sql = None
    
    # Look for SQL: marker
    sql_marker = re.search(r'SQL:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if sql_marker:
        sql = sql_marker.group(1).strip()
        # Remove markdown code blocks if present
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'```\s*$', '', sql)
        sql = sql.strip()
    
    # Look for ```sql code blocks
    if not sql:
        sql_block = re.search(r'```sql\s*(.*?)\s*```', text, re.IGNORECASE | re.DOTALL)
        if sql_block:
            sql = sql_block.group(1).strip()
    
    # Look for ``` blocks (generic)
    if not sql:
        code_block = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            sql = code_block.group(1).strip()
    
    # If no markers found, use entire text
    if not sql:
        sql = text.strip()
    
    # Validate that SQL starts with SELECT or WITH
    sql_upper = sql.upper().strip()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        raise ValueError(
            f"Extracted SQL does not start with SELECT or WITH. Got: {sql[:50]}... "
            "The model may have generated invalid output."
        )
    
    return sql


def generate_sql(
    prompt: str,
    model_name: str,
    temperature: float = 0.1,
    ollama_base_url: Optional[str] = None
) -> Tuple[str, Dict]:
    """
    Generate SQL from prompt using Ollama or OpenAI backend.
    Returns (sql_string, metrics_dict) where metrics_dict contains:
    - tokens_used: int (input + output tokens)
    - input_tokens: int
    - output_tokens: int
    - latency: float (seconds)
    """
    import time
    start_time = time.time()
    metrics = {
        "tokens_used": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency": 0.0
    }
    
    if model_name.startswith("ollama/"):
        model = model_name.replace("ollama/", "")
        if ollama_base_url is None:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        url = f"{ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            
            # Estimate tokens (rough: ~4 chars per token)
            metrics["input_tokens"] = len(prompt) // 4
            metrics["output_tokens"] = len(output) // 4
            metrics["tokens_used"] = metrics["input_tokens"] + metrics["output_tokens"]
            metrics["latency"] = time.time() - start_time
            
            try:
                sql = extract_sql_from_output(output)
                return sql, metrics
            except ValueError as ve:
                raise ValueError(f"Invalid SQL extraction: {str(ve)}")
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    elif model_name.startswith("openai/"):
        model = model_name.replace("openai/", "")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in environment")
        
        # Initialize OpenAI client - explicitly avoid proxy issues
        # The newer OpenAI library (1.0+) doesn't accept 'proxies' parameter
        # So we create client with minimal required parameters only
        try:
            # Create client with only api_key to avoid any proxy/environment conflicts
            client = OpenAI(api_key=api_key)
        except Exception as init_error:
            # If initialization fails, provide clearer error
            raise Exception(
                f"Failed to initialize OpenAI client: {str(init_error)}. "
                "Make sure you're using openai>=1.0.0. "
                "Try: pip install --upgrade openai"
            )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate valid SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                timeout=120
            )
            output = response.choices[0].message.content
            
            # Get actual token usage from OpenAI response
            if hasattr(response, 'usage'):
                metrics["input_tokens"] = response.usage.prompt_tokens
                metrics["output_tokens"] = response.usage.completion_tokens
                metrics["tokens_used"] = response.usage.total_tokens
            else:
                # Fallback estimation
                metrics["input_tokens"] = len(prompt) // 4
                metrics["output_tokens"] = len(output) // 4
                metrics["tokens_used"] = metrics["input_tokens"] + metrics["output_tokens"]
            
            metrics["latency"] = time.time() - start_time
            
            try:
                sql = extract_sql_from_output(output)
                return sql, metrics
            except ValueError as ve:
                raise ValueError(f"Invalid SQL extraction: {str(ve)}")
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    else:
        raise ValueError(f"Unknown model format: {model_name}. Use 'ollama/model' or 'openai/model'")


def refine_sql(
    refine_prompt: str,
    model_name: str,
    temperature: float = 0.2,
    ollama_base_url: Optional[str] = None
) -> Tuple[str, Dict]:
    """
    Refine SQL using execution-guided prompt.
    Wrapper around generate_sql with higher temperature for refinement.
    Returns (sql_string, metrics_dict).
    """
    return generate_sql(refine_prompt, model_name, temperature, ollama_base_url)

