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
    
    # For fine-tuned models, the output should be just SQL (from Response section)
    # But it might also include "SQL:" prefix or other text
    
    # Look for SQL: marker (might be at start of response)
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
    
    # If no markers found, use entire text (for fine-tuned models, this should be just SQL)
    if not sql:
        sql = text.strip()
    
    # Remove any leading "SQL:" if present (fine-tuned model might add it)
    sql = re.sub(r'^SQL:\s*', '', sql, flags=re.IGNORECASE).strip()
    
    # Remove trailing backticks and markdown formatting (common issue with base Arctic model)
    # Patterns: ```; or ``` or `; or ` at the end
    sql = re.sub(r'[`]+;?\s*$', '', sql).strip()
    sql = re.sub(r'```\s*;?\s*$', '', sql).strip()
    sql = re.sub(r'`+\s*$', '', sql).strip()
    
    # Ensure semicolon after cleaning (if SQL is valid)
    if sql and not sql.endswith(';'):
        sql_upper = sql.upper().strip()
        if sql_upper.startswith(('SELECT', 'WITH')):
            if not any(sql_upper.endswith(ending) for ending in ['LIMIT', 'DESC', 'ASC', ')']):
                sql = sql.rstrip() + ';'
    
    # Validate that SQL starts with SELECT or WITH
    sql_upper = sql.upper().strip()
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        raise ValueError(
            f"Extracted SQL does not start with SELECT or WITH. Got: {sql[:50]}... "
            "The model may have generated invalid output."
        )
    
    return sql


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
            # Set URL based on model name
            if "qwen-0.5b-base" in model_name.lower():
                # Qwen base model uses port 11439
                ollama_base_url = os.getenv("QWEN_BASE_SERVER_URL", "http://localhost:11439")
            elif "qwen" in model_name.lower():
                # Qwen finetuned model uses port 11438
                ollama_base_url = os.getenv("QWEN_SERVER_URL", "http://localhost:11438")
            else:
                # Default to Arctic server (port 11437)
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11437")
        
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
            # 2-minute timeout for all models (120 seconds)
            # This prevents hanging - if server doesn't respond, fail fast
            request_timeout = 120  # 2 minutes
            response = requests.post(url, json=payload, timeout=request_timeout)
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
                timeout=180 if "arctic" in model_name.lower() else 120  # Reduced timeout to prevent hanging
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
        raise ValueError(f"Unknown model format: {model_name}. Use 'ollama/arctic-base' or 'openai/gpt-4o-mini'")


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


async def generate_sql_async(
    prompt: str,
    model_name: str,
    temperature: float = 0.1,
    ollama_base_url: Optional[str] = None
) -> Tuple[str, Dict]:
    """
    Async version of generate_sql.
    """
    import time
    import aiohttp
    from openai import AsyncOpenAI
    
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
            # Set URL based on model name
            if "qwen-0.5b-base" in model_name.lower():
                # Qwen base model uses port 11439
                ollama_base_url = os.getenv("QWEN_BASE_SERVER_URL", "http://localhost:11439")
            elif "qwen" in model_name.lower():
                # Qwen finetuned model uses port 11438
                ollama_base_url = os.getenv("QWEN_SERVER_URL", "http://localhost:11438")
            else:
                # Default to Arctic server (port 11437)
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11437")
        
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=120) as response:
                    response.raise_for_status()
                    result = await response.json()
                    output = result.get("response", "")
            
            # Estimate tokens
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
            
        try:
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate valid SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                timeout=180 if "arctic" in model_name.lower() else 120  # Reduced timeout to prevent hanging
            )
            output = response.choices[0].message.content
            
            if hasattr(response, 'usage'):
                metrics["input_tokens"] = response.usage.prompt_tokens
                metrics["output_tokens"] = response.usage.completion_tokens
                metrics["tokens_used"] = response.usage.total_tokens
            else:
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
        raise ValueError(f"Unknown model format: {model_name}")


async def refine_sql_async(
    refine_prompt: str,
    model_name: str,
    temperature: float = 0.2,
    ollama_base_url: Optional[str] = None
) -> Tuple[str, Dict]:
    """Async version of refine_sql."""
    return await generate_sql_async(refine_prompt, model_name, temperature, ollama_base_url)

