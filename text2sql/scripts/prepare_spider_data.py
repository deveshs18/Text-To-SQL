"""Prepare Spider dataset for Qwen fine-tuning."""
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_tables(tables_path: str) -> Dict[str, str]:
    """
    Load tables.json and build schema strings for each database.
    Returns: {db_id: schema_string}
    """
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    schema_by_db = defaultdict(lambda: {"tables": defaultdict(list)})
    
    # Group columns by table for each database
    for table_info in tables:
        db_id = table_info["db_id"]
        table_names = table_info["table_names_original"]
        column_names = table_info["column_names_original"]
        
        # Process columns
        for col_idx, col_name in column_names:
            if col_name == "*":  # Skip wildcard
                continue
            if col_idx == -1:  # Special case
                continue
            
            table_idx = col_idx
            if table_idx < len(table_names):
                table_name = table_names[table_idx]
                schema_by_db[db_id]["tables"][table_name].append(col_name[1] if isinstance(col_name, list) else col_name)
    
    # Build schema strings: table1(col1, col2) | table2(col3, col4)
    schema_strings = {}
    for db_id, info in schema_by_db.items():
        parts = []
        for table_name, columns in sorted(info["tables"].items()):
            if columns:  # Only add if table has columns
                parts.append(f"{table_name}({', '.join(columns)})")
        schema_strings[db_id] = " | ".join(parts)
    
    return schema_strings


def format_training_example(question: str, sql: str, schema: str) -> str:
    """
    Format a training example in the format used by the existing system.
    Format: SCHEMA: ... Q: ... SQL: ...
    """
    return f"SCHEMA: {schema}\nQ: {question}\nSQL: {sql}"


def prepare_dataset(
    train_spider_path: str,
    train_others_path: str,
    tables_path: str,
    output_path: str
):
    """Prepare training dataset from Spider files."""
    print("=" * 80)
    print("PREPARING SPIDER DATASET FOR QWEN FINE-TUNING")
    print("=" * 80)
    
    # Load schema strings
    print("\n[1/4] Loading database schemas...")
    schema_by_db = load_tables(tables_path)
    print(f"✅ Loaded schemas for {len(schema_by_db)} databases")
    
    # Load training examples
    print("\n[2/4] Loading training examples...")
    with open(train_spider_path, 'r', encoding='utf-8') as f:
        train_spider = json.load(f)
    print(f"✅ Loaded {len(train_spider)} examples from train_spider.json")
    
    with open(train_others_path, 'r', encoding='utf-8') as f:
        train_others = json.load(f)
    print(f"✅ Loaded {len(train_others)} examples from train_others.json")
    
    all_train = train_spider + train_others
    print(f"✅ Total training examples: {len(all_train)}")
    
    # Format examples
    print("\n[3/4] Formatting training examples...")
    formatted_examples = []
    missing_schema = 0
    
    for ex in all_train:
        db_id = ex["db_id"]
        question = ex["question"].strip()
        sql = ex["query"].strip()
        
        schema = schema_by_db.get(db_id, "")
        if not schema:
            missing_schema += 1
            continue
        
        formatted = format_training_example(question, sql, schema)
        formatted_examples.append({
            "text": formatted,
            "db_id": db_id,
            "question": question,
            "sql": sql
        })
    
    if missing_schema:
        print(f"⚠️  Warning: {missing_schema} examples skipped (missing schema)")
    
    print(f"✅ Formatted {len(formatted_examples)} training examples")
    
    # Save to JSON
    print(f"\n[4/4] Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_examples, f, indent=2, ensure_ascii=False)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(formatted_examples)} examples ({file_size_mb:.2f} MB)")
    
    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE TRAINING EXAMPLE:")
    print("=" * 80)
    if formatted_examples:
        sample = formatted_examples[0]["text"]
        print(sample[:500] + "..." if len(sample) > 500 else sample)
    
    return formatted_examples


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    spider_dir = base_dir / "spider"
    
    train_spider_path = spider_dir / "train_spider.json"
    train_others_path = spider_dir / "train_others.json"
    tables_path = spider_dir / "tables.json"
    output_path = base_dir / "qwen_spider_train.json"
    
    # Check files exist
    for path in [train_spider_path, train_others_path, tables_path]:
        if not path.exists():
            print(f"❌ Error: {path} not found!")
            exit(1)
    
    # Prepare dataset
    prepare_dataset(
        str(train_spider_path),
        str(train_others_path),
        str(tables_path),
        str(output_path)
    )
    
    print("\n" + "=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print("Ready for training!")


