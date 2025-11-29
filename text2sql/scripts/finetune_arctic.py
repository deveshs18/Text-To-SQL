"""
Fine-tune Arctic-Text2SQL model with LoRA adapters.
This allows you to improve the model while preserving its excellent base performance.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os
import json
from datetime import datetime

print("="*80)
print("FINE-TUNING ARCTIC-TEXT2SQL MODEL")
print("="*80)
print("This will add LoRA adapters to improve the model while preserving base performance.")
print()

# Configuration
BASE_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"  # Base model for fine-tuning
OUTPUT_DIR = "arctic_lora_model"  # Your fine-tuned adapters
DATASET_PATH = "ft_spider/data/spider_alpaca_full.json"  # Spider dataset (8,659 examples)
TRAINING_SUBSET_SIZE = 500  # Use only 500 examples for quick fine-tuning (change to None for full dataset)

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available. Fine-tuning requires GPU.")
    exit(1)

print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# 1. Load Base Model (Arctic-Text2SQL)
print("[1/7] Loading base model: Arctic-Text2SQL-R1-7B")
print("This may take a few minutes (downloading if first time)...")

# 4-bit quantization for memory efficiency
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
tokenizer.padding_side = "right"

print("✅ Base model loaded!")

# 2. Prepare Model for Training
print("\n[2/7] Preparing model for training...")
model = prepare_model_for_kbit_training(model)

# 3. Add LoRA Adapters
print("\n[3/7] Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,  # Rank (lower = smaller adapters, less overfitting)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}%")

# 4. Load Dataset
print("\n[4/7] Loading dataset...")
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found: {DATASET_PATH}")
    print("💡 Run prepare_full_spider.py first to create the dataset")
    exit(1)

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"✅ Loaded {len(dataset)} examples")

# Use subset for quick fine-tuning (if specified)
if TRAINING_SUBSET_SIZE and TRAINING_SUBSET_SIZE < len(dataset):
    print(f"📊 Using subset of {TRAINING_SUBSET_SIZE} examples for quick fine-tuning")
    dataset = dataset.select(range(TRAINING_SUBSET_SIZE))
    print(f"✅ Using {len(dataset)} examples for training")

# 5. Format Dataset
print("\n[5/7] Formatting dataset...")
def format_instruction(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    # Format in Alpaca style (matches Arctic's training format)
    text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}
SQL:

### Response:
{output}"""
    
    return {"text": text}

dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
print("✅ Dataset formatted!")

# 6. Training Arguments
print("\n[6/7] Setting up training...")

# Conservative training to preserve base performance
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Start with 1 epoch (can increase if needed)
    per_device_train_batch_size=2,  # Small batch for stability
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=50,
    learning_rate=1e-4,  # Lower LR to preserve base performance
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="no",
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
    gradient_checkpointing=True,  # Save memory
    dataloader_num_workers=0,  # Windows compatibility
    max_grad_norm=1.0,
)

# 7. Create Trainer
print("\n[7/7] Creating trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    dataset_text_field="text",  # Field name in formatted dataset
    max_seq_length=1536,
)

# Training Summary
print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"Base Model: {BASE_MODEL}")
print(f"Dataset: {len(dataset)} examples")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Max sequence length: 1536")
print()
print("💡 This fine-tuning will:")
print("   - Preserve the excellent base performance of Arctic")
print("   - Add your own improvements via LoRA adapters")
print("   - Allow you to legitimately claim you trained/improved the model")
print("="*80)
print()

# Start Training
print("Starting training...")
print("="*80)
trainer.train()

# Save Model
print("\n" + "="*80)
print("SAVING FINE-TUNED MODEL")
print("="*80)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training summary
summary = {
    "base_model": BASE_MODEL,
    "fine_tuning_method": "LoRA",
    "dataset_size": len(dataset),
    "epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "training_date": datetime.now().isoformat(),
    "trainable_params": trainable_params,
    "total_params": all_params,
    "trainable_percentage": f"{100 * trainable_params / all_params:.2f}%"
}

with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Fine-tuned model saved to: {OUTPUT_DIR}/")
print(f"✅ Training summary saved to: {OUTPUT_DIR}/training_summary.json")
print()
print("="*80)
print("TRAINING COMPLETE")
print("="*80)
print()
print("Next steps:")
print("1. Test the fine-tuned model")
print("2. Compare performance with base Arctic model")
print("3. If improved, you can legitimately say:")
print("   'I fine-tuned the Arctic-Text2SQL model using LoRA adapters'")
print("   'I trained an improved version of the model on the Spider dataset'")
print("="*80)

