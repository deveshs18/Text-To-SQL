"""RAG-lite schema retrieval: compact schema snippets from database introspection."""
from typing import List, Tuple
from sqlalchemy import create_engine, text, inspect


def get_schema_snippet(question: str, engine, table_name: str = "adult_income") -> str:
    """
    Get compact schema snippet for a table.
    Returns 1-3 line description with column names.
    """
    try:
        # SQLite introspection
        with engine.connect() as conn:
            # Get column info
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = result.fetchall()
            
            if not columns:
                return f"{table_name}(...)"
            
            # Extract column names (index 1 in PRAGMA table_info result)
            col_names = [col[1] for col in columns]
            
            # Build compact snippet
            cols_str = ", ".join(col_names)
            
            # Single line if short, multi-line if long
            if len(cols_str) < 80:
                snippet = f"{table_name}({cols_str})"
            else:
                # Split into reasonable chunks
                lines = []
                current_line = f"{table_name}(\n  "
                current_len = len(current_line)
                
                for col in col_names:
                    col_with_comma = col + ", "
                    if current_len + len(col_with_comma) > 75:
                        lines.append(current_line.rstrip(", "))
                        current_line = "  " + col_with_comma
                        current_len = len(current_line)
                    else:
                        current_line += col_with_comma
                        current_len += len(col_with_comma)
                
                if current_line.strip():
                    lines.append(current_line.rstrip(", "))
                lines.append(")")
                snippet = "\n".join(lines)
            
            # Optional: add brief hints for key columns
            hints = []
            if "income" in col_names:
                hints.append("income ∈ {'<=50K', '>50K'}")
            if "education" in col_names:
                hints.append("education: various levels (e.g., 'Bachelors', 'HS-grad')")
            
            if hints:
                snippet += "\n" + " | ".join(hints)
            
            return snippet
            
    except Exception as e:
        return f"{table_name}(...columns unavailable: {str(e)})"


def get_all_tables(engine) -> List[str]:
    """Get list of all table names in the database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception:
        # Fallback for SQLite
        try:
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ))
                return [row[0] for row in result.fetchall()]
        except Exception:
            return []


def expand_schema_snippet(engine, table_name: str = "adult_income", peek_values: bool = True) -> str:
    """
    Get expanded schema with more details (for fallback when initial prompts fail).
    Optionally peek at distinct values for a few key columns.
    """
    base_snippet = get_schema_snippet("", engine, table_name)
    
    if not peek_values:
        return base_snippet
    
    try:
        with engine.connect() as conn:
            # Peek at distinct values for income column if present
            try:
                result = conn.execute(text(
                    f"SELECT DISTINCT income FROM {table_name} LIMIT 10"
                ))
                income_vals = [row[0] for row in result.fetchall() if row[0]]
                if income_vals:
                    base_snippet += f"\nDistinct income values: {income_vals[:5]}"
            except:
                pass
            
            # Peek at education if present
            try:
                result = conn.execute(text(
                    f"SELECT DISTINCT education FROM {table_name} LIMIT 10"
                ))
                edu_vals = [row[0] for row in result.fetchall() if row[0]]
                if edu_vals:
                    base_snippet += f"\nSample education values: {edu_vals[:5]}"
            except:
                pass
                
    except Exception:
        pass
    
    return base_snippet


