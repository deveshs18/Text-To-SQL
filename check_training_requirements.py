"""Check all requirements for training."""
import sys

print("="*60)
print("CHECKING TRAINING REQUIREMENTS")
print("="*60)

all_ok = True

# Check PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA: Available ({torch.cuda.get_device_name(0)})")
    else:
        print("❌ CUDA: NOT AVAILABLE")
        all_ok = False
except ImportError:
    print("❌ PyTorch: NOT INSTALLED")
    all_ok = False

# Check transformers
try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
except ImportError:
    print("❌ transformers: NOT INSTALLED")
    all_ok = False

# Check peft
try:
    import peft
    print(f"✅ peft: {peft.__version__}")
except ImportError:
    print("❌ peft: NOT INSTALLED")
    all_ok = False

# Check trl
try:
    import trl
    print(f"✅ trl: {trl.__version__}")
except ImportError:
    print("❌ trl: NOT INSTALLED")
    all_ok = False

# Check bitsandbytes
try:
    import bitsandbytes
    print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("❌ bitsandbytes: NOT INSTALLED")
    all_ok = False

# Check datasets
try:
    import datasets
    print(f"✅ datasets: {datasets.__version__}")
except ImportError:
    print("❌ datasets: NOT INSTALLED")
    all_ok = False

# Check dataset file
import os
dataset_path = "ft_spider/data/spider_alpaca_full.json"
if os.path.exists(dataset_path):
    size_mb = os.path.getsize(dataset_path) / (1024*1024)
    print(f"✅ Dataset: Found ({size_mb:.2f} MB)")
    
    # Check dataset content
    import json
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Dataset: {len(data)} examples")
else:
    print(f"❌ Dataset: NOT FOUND at {dataset_path}")
    all_ok = False

# Check training script
script_path = "text2sql/scripts/finetune_arctic.py"
if os.path.exists(script_path):
    print(f"✅ Training script: Found")
else:
    print(f"❌ Training script: NOT FOUND at {script_path}")
    all_ok = False

print("="*60)
if all_ok:
    print("✅ ALL REQUIREMENTS MET - READY TO TRAIN!")
else:
    print("❌ SOME REQUIREMENTS MISSING - SEE ABOVE")
print("="*60)

sys.exit(0 if all_ok else 1)

