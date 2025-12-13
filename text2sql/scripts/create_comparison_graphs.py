"""
Create comparison graphs for model evaluation:
1. Qwen 0.5 Base vs Finetuned - EX comparison (RAG)
2. Qwen 0.5 Base vs Finetuned - Success Rate comparison (RAG)
3. Qwen 0.5 Base vs Finetuned - Latency comparison
4. Qwen 7B (Arctic) vs GPT vs Qwen 0.5 Finetuned - EX & Success Rate
5. Qwen 7B (Arctic) vs GPT vs Qwen 0.5 Finetuned - Latency comparison
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Setup paths
script_dir = Path(__file__).parent
results_dir = script_dir / "data" / "results"
output_dir = results_dir / "presentation"
output_dir.mkdir(parents=True, exist_ok=True)

# Configuration
TECHNIQUES = ['Few-Shot', 'CoT', 'LtM', 'EG']

# Model mappings
MODEL_PREFIXES = {
    'qwen_base': 'ollama_qwen-0.5b-base',
    'qwen_05_finetuned': 'ollama_qwen-0.5b-spider',
    'qwen_7b_finetuned': 'ollama_arctic-base',  # Arctic is actually Qwen 7B Finetuned
    'gpt': 'openai_gpt-4o-mini'
}

MODEL_DISPLAY_NAMES = {
    'qwen_base': 'Qwen 0.5 Base',
    'qwen_05_finetuned': 'Qwen 0.5 Finetuned',
    'qwen_7b_finetuned': 'Qwen 7B Finetuned',
    'gpt': 'GPT-4o-mini'
}

# Color scheme
COLORS = {
    'qwen_base': '#FF6B6B',      # Red
    'qwen_05_finetuned': '#4ECDC4',  # Teal
    'qwen_7b_finetuned': '#45B7D1',  # Blue
    'gpt': '#FFA07A',            # Light Salmon
    'with_rag': '#2E86AB',        # Blue
    'without_rag': '#E63946',     # Red
}

def load_model_results(model_prefix):
    """Load all technique results for a model."""
    dfs = []
    for technique in TECHNIQUES:
        csv_path = results_dir / f"{model_prefix}_{technique}_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
    
    if not dfs:
        return None
    
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def load_technique_results(model_prefix, technique):
    """Load results for a specific model and technique."""
    csv_path = results_dir / f"{model_prefix}_{technique}_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

def calculate_metrics(df):
    """Calculate EX, Success Rate, and Latency metrics."""
    total = len(df)
    
    # Execution Accuracy (EX)
    with_rag_ex = (df['With_RAG_EX'] == '✅').sum()
    without_rag_ex = (df['Without_RAG_EX'] == '✅').sum()
    
    # Success Rate
    with_rag_success = (df['With_RAG_Success'] == '✅').sum()
    without_rag_success = (df['Without_RAG_Success'] == '✅').sum()
    
    # Latency
    with_rag_latencies = []
    without_rag_latencies = []
    
    for lat_str in df['With_RAG_Latency']:
        if lat_str not in ['SKIPPED', 'TIMEOUT']:
            try:
                lat_val = float(str(lat_str).replace('s', '').strip())
                with_rag_latencies.append(lat_val)
            except:
                pass
    
    for lat_str in df['Without_RAG_Latency']:
        if lat_str not in ['SKIPPED', 'TIMEOUT']:
            try:
                lat_val = float(str(lat_str).replace('s', '').strip())
                without_rag_latencies.append(lat_val)
            except:
                pass
    
    avg_with_rag_latency = np.mean(with_rag_latencies) if with_rag_latencies else 0
    avg_without_rag_latency = np.mean(without_rag_latencies) if without_rag_latencies else 0
    
    return {
        'total': total,
        'with_rag_ex': with_rag_ex,
        'without_rag_ex': without_rag_ex,
        'with_rag_ex_pct': (with_rag_ex / total * 100) if total > 0 else 0,
        'without_rag_ex_pct': (without_rag_ex / total * 100) if total > 0 else 0,
        'with_rag_success': with_rag_success,
        'without_rag_success': without_rag_success,
        'with_rag_success_pct': (with_rag_success / total * 100) if total > 0 else 0,
        'without_rag_success_pct': (without_rag_success / total * 100) if total > 0 else 0,
        'avg_with_rag_latency': avg_with_rag_latency,
        'avg_without_rag_latency': avg_without_rag_latency,
    }

def create_graph_1_qwen_base_vs_finetuned_ex():
    """Graph 1: Qwen 0.5 Base vs Finetuned - EX Comparison (With/Without RAG)"""
    print("\n[Graph 1] Creating Qwen Base vs Finetuned - EX Comparison...")
    
    base_df = load_model_results(MODEL_PREFIXES['qwen_base'])
    finetuned_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    
    if base_df is None or finetuned_df is None:
        print("⚠️  Missing data for Graph 1")
        return None
    
    base_metrics = calculate_metrics(base_df)
    finetuned_metrics = calculate_metrics(finetuned_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Qwen 0.5 Base', 'Qwen 0.5 Finetuned']
    with_rag = [base_metrics['with_rag_ex_pct'], finetuned_metrics['with_rag_ex_pct']]
    without_rag = [base_metrics['without_rag_ex_pct'], finetuned_metrics['without_rag_ex_pct']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_rag, width, label='With RAG', color=COLORS['with_rag'], alpha=0.8)
    bars2 = ax.bar(x + width/2, without_rag, width, label='Without RAG', color=COLORS['without_rag'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Accuracy (EX) %', fontsize=12, fontweight='bold')
    ax.set_title('Qwen 0.5 Base vs Finetuned - Execution Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(with_rag), max(without_rag)) * 1.2)
    
    plt.tight_layout()
    output_path = output_dir / 'graph1_qwen_base_vs_finetuned_ex.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_graph_2_qwen_base_vs_finetuned_success():
    """Graph 2: Qwen 0.5 Base vs Finetuned - Success Rate Comparison (With/Without RAG)"""
    print("\n[Graph 2] Creating Qwen Base vs Finetuned - Success Rate Comparison...")
    
    base_df = load_model_results(MODEL_PREFIXES['qwen_base'])
    finetuned_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    
    if base_df is None or finetuned_df is None:
        print("⚠️  Missing data for Graph 2")
        return None
    
    base_metrics = calculate_metrics(base_df)
    finetuned_metrics = calculate_metrics(finetuned_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Qwen 0.5 Base', 'Qwen 0.5 Finetuned']
    with_rag = [base_metrics['with_rag_success_pct'], finetuned_metrics['with_rag_success_pct']]
    without_rag = [base_metrics['without_rag_success_pct'], finetuned_metrics['without_rag_success_pct']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_rag, width, label='With RAG', color=COLORS['with_rag'], alpha=0.8)
    bars2 = ax.bar(x + width/2, without_rag, width, label='Without RAG', color=COLORS['without_rag'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate %', fontsize=12, fontweight='bold')
    ax.set_title('Qwen 0.5 Base vs Finetuned - Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = output_dir / 'graph2_qwen_base_vs_finetuned_success.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_graph_3_qwen_base_vs_finetuned_latency():
    """Graph 3: Qwen 0.5 Base vs Finetuned - Latency Comparison"""
    print("\n[Graph 3] Creating Qwen Base vs Finetuned - Latency Comparison...")
    
    base_df = load_model_results(MODEL_PREFIXES['qwen_base'])
    finetuned_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    
    if base_df is None or finetuned_df is None:
        print("⚠️  Missing data for Graph 3")
        return None
    
    base_metrics = calculate_metrics(base_df)
    finetuned_metrics = calculate_metrics(finetuned_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Qwen 0.5 Base', 'Qwen 0.5 Finetuned']
    with_rag = [base_metrics['avg_with_rag_latency'], finetuned_metrics['avg_with_rag_latency']]
    without_rag = [base_metrics['avg_without_rag_latency'], finetuned_metrics['avg_without_rag_latency']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_rag, width, label='With RAG', color=COLORS['with_rag'], alpha=0.8)
    bars2 = ax.bar(x + width/2, without_rag, width, label='Without RAG', color=COLORS['without_rag'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Qwen 0.5 Base vs Finetuned - Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(with_rag), max(without_rag)) * 1.2)
    
    plt.tight_layout()
    output_path = output_dir / 'graph3_qwen_base_vs_finetuned_latency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_graph_4_all_models_ex_success():
    """Graph 4: Qwen 7B Finetuned vs GPT vs Qwen 0.5 Finetuned - EX & Success Rate"""
    print("\n[Graph 4] Creating All Models - EX & Success Rate Comparison...")
    
    qwen_7b_df = load_model_results(MODEL_PREFIXES['qwen_7b_finetuned'])
    gpt_df = load_model_results(MODEL_PREFIXES['gpt'])
    qwen_05_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    
    if qwen_7b_df is None or gpt_df is None or qwen_05_df is None:
        print("⚠️  Missing data for Graph 4")
        return None
    
    qwen_7b_metrics = calculate_metrics(qwen_7b_df)
    gpt_metrics = calculate_metrics(gpt_df)
    qwen_05_metrics = calculate_metrics(qwen_05_df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = ['Qwen 7B\nFinetuned', 'GPT-4o-mini', 'Qwen 0.5\nFinetuned']
    ex_with_rag = [qwen_7b_metrics['with_rag_ex_pct'], gpt_metrics['with_rag_ex_pct'], qwen_05_metrics['with_rag_ex_pct']]
    success_with_rag = [qwen_7b_metrics['with_rag_success_pct'], gpt_metrics['with_rag_success_pct'], qwen_05_metrics['with_rag_success_pct']]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Left subplot: Execution Accuracy
    bars1 = ax1.bar(x, ex_with_rag, width, color=[COLORS['qwen_7b_finetuned'], COLORS['gpt'], COLORS['qwen_05_finetuned']], alpha=0.8)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Accuracy (EX) %', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Accuracy (With RAG)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Right subplot: Success Rate
    bars2 = ax2.bar(x, success_with_rag, width, color=[COLORS['qwen_7b_finetuned'], COLORS['gpt'], COLORS['qwen_05_finetuned']], alpha=0.8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate %', fontsize=12, fontweight='bold')
    ax2.set_title('Success Rate (With RAG)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = output_dir / 'graph4_all_models_ex_success.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_graph_5_all_models_latency():
    """Graph 5: Qwen 7B Finetuned vs GPT vs Qwen 0.5 Finetuned - Latency Comparison"""
    print("\n[Graph 5] Creating All Models - Latency Comparison...")
    
    qwen_7b_df = load_model_results(MODEL_PREFIXES['qwen_7b_finetuned'])
    gpt_df = load_model_results(MODEL_PREFIXES['gpt'])
    qwen_05_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    
    if qwen_7b_df is None or gpt_df is None or qwen_05_df is None:
        print("⚠️  Missing data for Graph 5")
        return None
    
    qwen_7b_metrics = calculate_metrics(qwen_7b_df)
    gpt_metrics = calculate_metrics(gpt_df)
    qwen_05_metrics = calculate_metrics(qwen_05_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Qwen 7B\nFinetuned', 'GPT-4o-mini', 'Qwen 0.5\nFinetuned']
    with_rag = [qwen_7b_metrics['avg_with_rag_latency'], gpt_metrics['avg_with_rag_latency'], qwen_05_metrics['avg_with_rag_latency']]
    without_rag = [qwen_7b_metrics['avg_without_rag_latency'], gpt_metrics['avg_without_rag_latency'], qwen_05_metrics['avg_without_rag_latency']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_rag, width, label='With RAG', color=COLORS['with_rag'], alpha=0.8)
    bars2 = ax.bar(x + width/2, without_rag, width, label='Without RAG', color=COLORS['without_rag'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('All Models - Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(with_rag), max(without_rag)) * 1.2)
    
    plt.tight_layout()
    output_path = output_dir / 'graph5_all_models_latency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_graph_6_comprehensive_comparison():
    """Graph 6: Comprehensive comparison - EX, Success Rate, Latency (all models)"""
    print("\n[Graph 6] Creating Comprehensive Comparison (EX, Success, Latency)...")
    
    qwen_7b_df = load_model_results(MODEL_PREFIXES['qwen_7b_finetuned'])
    gpt_df = load_model_results(MODEL_PREFIXES['gpt'])
    qwen_05_df = load_model_results(MODEL_PREFIXES['qwen_05_finetuned'])
    base_df = load_model_results(MODEL_PREFIXES['qwen_base'])
    
    if not all([qwen_7b_df is not None, gpt_df is not None, qwen_05_df is not None, base_df is not None]):
        print("⚠️  Missing data for Graph 6")
        return None
    
    qwen_7b_metrics = calculate_metrics(qwen_7b_df)
    gpt_metrics = calculate_metrics(gpt_df)
    qwen_05_metrics = calculate_metrics(qwen_05_df)
    base_metrics = calculate_metrics(base_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['Qwen 0.5\nBase', 'Qwen 0.5\nFinetuned', 'Qwen 7B\nFinetuned', 'GPT-4o-mini']
    ex_with_rag = [base_metrics['with_rag_ex_pct'], qwen_05_metrics['with_rag_ex_pct'], 
                   qwen_7b_metrics['with_rag_ex_pct'], gpt_metrics['with_rag_ex_pct']]
    success_with_rag = [base_metrics['with_rag_success_pct'], qwen_05_metrics['with_rag_success_pct'],
                        qwen_7b_metrics['with_rag_success_pct'], gpt_metrics['with_rag_success_pct']]
    latency_with_rag = [base_metrics['avg_with_rag_latency'], qwen_05_metrics['avg_with_rag_latency'],
                        qwen_7b_metrics['avg_with_rag_latency'], gpt_metrics['avg_with_rag_latency']]
    
    colors = [COLORS['qwen_base'], COLORS['qwen_05_finetuned'], COLORS['qwen_7b_finetuned'], COLORS['gpt']]
    x = np.arange(len(models))
    width = 0.7
    
    # EX
    bars1 = axes[0].bar(x, ex_with_rag, width, color=colors, alpha=0.8)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Execution Accuracy (EX) %', fontsize=11, fontweight='bold')
    axes[0].set_title('Execution Accuracy (With RAG)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # Success Rate
    bars2 = axes[1].bar(x, success_with_rag, width, color=colors, alpha=0.8)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Success Rate %', fontsize=11, fontweight='bold')
    axes[1].set_title('Success Rate (With RAG)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    # Latency
    bars3 = axes[2].bar(x, latency_with_rag, width, color=colors, alpha=0.8)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Average Latency (seconds)', fontsize=11, fontweight='bold')
    axes[2].set_title('Latency (With RAG)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=10)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim(0, max(latency_with_rag) * 1.2)
    
    plt.tight_layout()
    output_path = output_dir / 'graph6_comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path

def create_technique_graphs():
    """Create graphs for each prompt technique showing 3 models."""
    print("\n" + "="*80)
    print("CREATING TECHNIQUE-SPECIFIC GRAPHS")
    print("="*80)
    
    graph_paths = []
    
    # Models to compare (3 models)
    models_to_compare = {
        'qwen_05_finetuned': {
            'prefix': MODEL_PREFIXES['qwen_05_finetuned'],
            'name': MODEL_DISPLAY_NAMES['qwen_05_finetuned'],
            'color': COLORS['qwen_05_finetuned']
        },
        'qwen_7b_finetuned': {
            'prefix': MODEL_PREFIXES['qwen_7b_finetuned'],
            'name': MODEL_DISPLAY_NAMES['qwen_7b_finetuned'],
            'color': COLORS['qwen_7b_finetuned']
        },
        'gpt': {
            'prefix': MODEL_PREFIXES['gpt'],
            'name': MODEL_DISPLAY_NAMES['gpt'],
            'color': COLORS['gpt']
        }
    }
    
    for technique in TECHNIQUES:
        print(f"\n[Technique: {technique}] Creating graphs...")
        
        # Load data for each model
        model_data = {}
        for model_key, model_info in models_to_compare.items():
            df = load_technique_results(model_info['prefix'], technique)
            if df is not None:
                model_data[model_key] = {
                    'metrics': calculate_metrics(df),
                    'name': model_info['name'],
                    'color': model_info['color']
                }
        
        if len(model_data) < 3:
            print(f"⚠️  Missing data for {technique} - skipping")
            continue
        
        # Create graph for this technique: EX, Success Rate, Latency
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = [model_data[k]['name'] for k in models_to_compare.keys() if k in model_data]
        colors_list = [model_data[k]['color'] for k in models_to_compare.keys() if k in model_data]
        
        ex_with_rag = [model_data[k]['metrics']['with_rag_ex_pct'] for k in models_to_compare.keys() if k in model_data]
        success_with_rag = [model_data[k]['metrics']['with_rag_success_pct'] for k in models_to_compare.keys() if k in model_data]
        latency_with_rag = [model_data[k]['metrics']['avg_with_rag_latency'] for k in models_to_compare.keys() if k in model_data]
        
        x = np.arange(len(model_names))
        width = 0.7
        
        # EX
        bars1 = axes[0].bar(x, ex_with_rag, width, color=colors_list, alpha=0.8)
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Execution Accuracy (EX) %', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Execution Accuracy - {technique}', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Success Rate
        bars2 = axes[1].bar(x, success_with_rag, width, color=colors_list, alpha=0.8)
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Success Rate %', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Success Rate - {technique}', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # Latency
        bars3 = axes[2].bar(x, latency_with_rag, width, color=colors_list, alpha=0.8)
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Latency - {technique}', fontsize=13, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, fontsize=10)
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].set_ylim(0, max(latency_with_rag) * 1.2 if latency_with_rag else 1)
        
        plt.tight_layout()
        technique_safe = technique.replace('-', '_').replace(' ', '_')
        output_path = output_dir / f'technique_{technique_safe}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        graph_paths.append(output_path)
    
    return graph_paths

def main():
    """Main function to create all graphs."""
    print("="*80)
    print("CREATING COMPARISON GRAPHS")
    print("="*80)
    print("\nGraphs to create:")
    print("  1. Qwen 0.5 Base vs Finetuned - EX Comparison")
    print("  2. Qwen 0.5 Base vs Finetuned - Success Rate Comparison")
    print("  3. Qwen 0.5 Base vs Finetuned - Latency Comparison")
    print("  4. Qwen 7B Finetuned vs GPT vs Qwen 0.5 Finetuned - EX & Success Rate")
    print("  5. Qwen 7B Finetuned vs GPT vs Qwen 0.5 Finetuned - Latency")
    print("  6. Comprehensive Comparison - All Models (EX, Success, Latency)")
    print("  7-10. Technique-specific graphs (Few-Shot, CoT, LtM, EG)")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # Create all graphs
    graph_paths = []
    
    graph_paths.append(create_graph_1_qwen_base_vs_finetuned_ex())
    graph_paths.append(create_graph_2_qwen_base_vs_finetuned_success())
    graph_paths.append(create_graph_3_qwen_base_vs_finetuned_latency())
    graph_paths.append(create_graph_4_all_models_ex_success())
    graph_paths.append(create_graph_5_all_models_latency())
    graph_paths.append(create_graph_6_comprehensive_comparison())
    
    # Create technique-specific graphs
    technique_graphs = create_technique_graphs()
    graph_paths.extend(technique_graphs)
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL GRAPHS CREATED!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\nCreated graphs:")
    for i, path in enumerate(graph_paths, 1):
        if path:
            print(f"  {i}. {path.name}")
    print("="*80)

if __name__ == "__main__":
    main()

