"""Fine-tune Qwen2.5-Coder-0.5B on Spider dataset using QLoRA."""
import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset


def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU memory cleared")


def load_dataset_from_json(json_path: str):
    """Load dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to format expected by datasets library
    texts = [item["text"] for item in data]
    return Dataset.from_dict({"text": texts})


def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize examples and create labels that mask prompt tokens (only compute loss on SQL part)."""
    texts = examples["text"]
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for text in texts:
        # Find where SQL starts (after "SQL:")
        sql_marker = "SQL:"
        sql_start_idx = text.find(sql_marker)
        
        if sql_start_idx == -1:
            # If no SQL marker, mask everything (shouldn't happen with our data)
            prompt_text = text
        else:
            # Everything up to and including "SQL:" is the prompt
            prompt_text = text[:sql_start_idx + len(sql_marker)]
        
        # Tokenize prompt separately to find its length in token space
        prompt_tokens = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors=None
        )
        prompt_token_len = len(prompt_tokens["input_ids"])
        
        # Tokenize full text for input_ids
        full_tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels: -100 (masked) for prompt, actual token IDs for SQL
        labels = [-100] * max_length
        
        # Copy SQL tokens to labels (everything after prompt)
        full_token_len = len(full_tokens["input_ids"])
        sql_start = min(prompt_token_len, max_length)
        sql_end = min(full_token_len, max_length)
        
        for i in range(sql_start, sql_end):
            if i < len(full_tokens["input_ids"]):
                labels[i] = full_tokens["input_ids"][i]
        
        input_ids_list.append(full_tokens["input_ids"])
        attention_mask_list.append(full_tokens["attention_mask"])
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


def main():
    print("=" * 80)
    print("QWEN2.5-CODER-0.5B FINE-TUNING ON SPIDER DATASET")
    print("=" * 80)
    
    # Clear GPU memory first
    print("\n[1/7] Clearing GPU memory...")
    clear_gpu_memory()
    
    # Check GPU
    print("\n[2/7] Checking GPU...")
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training requires GPU.")
        return
    
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {device_name}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    dataset_path = base_dir / "qwen_spider_train.json"
    output_dir = base_dir / "qwen0p5b-spider-lora"
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please run prepare_spider_data.py first!")
        return
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    print(f"\n[3/7] Loading model: {model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    print("   Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    print("\n[4/7] Configuring LoRA adapters...")
    # For Qwen models, check the actual module names
    # Common names: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Will verify/adjust if needed
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load tokenizer
    print("\n[5/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    print("\n[6/7] Loading dataset...")
    dataset = load_dataset_from_json(str(dataset_path))
    print(f"✅ Loaded {len(dataset)} training examples")
    
    # Tokenize dataset
    print("   Tokenizing dataset...")
    MAX_LENGTH = 1024
    
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer, max_length=MAX_LENGTH)
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Training arguments
    print("\n[7/7] Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard for now
        remove_unused_columns=False,
    )
    
    # Create trainer
    # Labels are already set in tokenize_function to mask prompt tokens
    # Only SQL tokens will have loss computed (tokens after "SQL:")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        args=training_args,
        max_seq_length=MAX_LENGTH,
        packing=False,
        dataset_text_field="text",
    )
    
    # Calculate training time estimate
    num_examples = len(tokenized_dataset)
    steps_per_epoch = num_examples // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION:")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Training examples: {num_examples:,}")
    print(f"Batch size (per device): {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Total steps: ~{total_steps:,}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80)
    print("This may take 12-18 hours on RTX 4060 Laptop GPU")
    print("Monitor progress with logging_steps=50")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING MODEL...")
    print("=" * 80)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✅ Model saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

