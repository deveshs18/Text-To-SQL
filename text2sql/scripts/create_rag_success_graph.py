"""
Create a graph showing which prompts worked with RAG across all 3 models.
Shows success rate (EX with RAG) for each model-technique combination.
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
MODEL_NAMES_MAP = {
    'ollama_qwen-0.5b-spider': 'Qwen-0.5B',
    'ollama_arctic-base': 'Qwen 7B',
    'openai_gpt-4o-mini': 'GPT-4o-mini'
}

def load_all_results():
    """Load all result files and calculate success rates."""
    results_data = {}
    
    for model_prefix, display_name in MODEL_NAMES_MAP.items():
        results_data[display_name] = {}
        
        for technique in TECHNIQUES:
            file_path = results_dir / f"{model_prefix}_{technique}_results.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Filter out skipped cases
                df_filtered = df[df['With_RAG_Success'] != 'SKIPPED']
                
                # Count total queries
                total_queries = len(df_filtered)
                
                # Count queries that passed EX with RAG
                passed_ex_rag = sum(df_filtered['With_RAG_EX'] == '✅')
                
                # Calculate success rate
                success_rate = (passed_ex_rag / total_queries * 100) if total_queries > 0 else 0
                
                results_data[display_name][technique] = {
                    'passed': passed_ex_rag,
                    'total': total_queries,
                    'success_rate': success_rate
                }
            else:
                results_data[display_name][technique] = {
                    'passed': 0,
                    'total': 0,
                    'success_rate': 0
                }
    
    return results_data

def create_heatmap_graph(results_data):
    """Create a heatmap showing success rates."""
    # Prepare data for heatmap
    models = list(MODEL_NAMES_MAP.values())
    techniques = TECHNIQUES
    
    # Create matrix
    matrix = np.zeros((len(techniques), len(models)))
    annotation_matrix = []
    
    for i, technique in enumerate(techniques):
        row = []
        for j, model in enumerate(models):
            data = results_data[model][technique]
            matrix[i, j] = data['success_rate']
            # Annotation: Show only percentage
            annotation = f"{data['success_rate']:.1f}%"
            row.append(annotation)
        annotation_matrix.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=annotation_matrix,
        fmt='',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Execution Accuracy (%)'},
        xticklabels=models,
        yticklabels=techniques,
        linewidths=1,
        linecolor='white',
        ax=ax
    )
    
    ax.set_title('Execution Accuracy (EX) with RAG: Model vs Prompt Technique', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Technique', fontsize=12, fontweight='bold')
    # Update colorbar label to emphasize "Accuracy"
    cbar = ax.collections[0].colorbar
    cbar.set_label('Execution Accuracy (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "rag_success_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap saved: {output_path}")
    return output_path

def create_bar_chart(results_data):
    """Create a grouped bar chart showing success rates."""
    models = list(MODEL_NAMES_MAP.values())
    techniques = TECHNIQUES
    
    # Prepare data
    x = np.arange(len(techniques))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars for each model
    for i, model in enumerate(models):
        success_rates = [results_data[model][tech]['success_rate'] for tech in techniques]
        bars = ax.bar(x + i * width, success_rates, width, label=model, alpha=0.8)
        
        # Add value labels on bars (percentage only)
        for j, (bar, rate) in enumerate(zip(bars, success_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Accuracy with RAG: Comparison Across Models and Techniques', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(techniques)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "rag_success_barchart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Bar chart saved: {output_path}")
    return output_path

def create_summary_table(results_data):
    """Create a summary table."""
    models = list(MODEL_NAMES_MAP.values())
    techniques = TECHNIQUES
    
    print("\n" + "="*80)
    print("EXECUTION ACCURACY (EX) WITH RAG - SUMMARY")
    print("="*80)
    print(f"{'Technique':<15} {'Qwen-0.5B':<20} {'Qwen 7B':<20} {'GPT-4o-mini':<20}")
    print("-"*80)
    
    for technique in techniques:
        row = [technique]
        for model in models:
            data = results_data[model][technique]
            row.append(f"{data['passed']}/{data['total']} ({data['success_rate']:.1f}%)")
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    print("="*80)
    
    # Calculate totals
    print("\nTOTAL ACROSS ALL TECHNIQUES:")
    print("-"*80)
    for model in models:
        total_passed = sum(results_data[model][tech]['passed'] for tech in techniques)
        total_queries = sum(results_data[model][tech]['total'] for tech in techniques)
        total_rate = (total_passed / total_queries * 100) if total_queries > 0 else 0
        print(f"{model}: {total_passed}/{total_queries} ({total_rate:.1f}%)")
    print("="*80)

def main():
    print("="*80)
    print("CREATING RAG SUCCESS GRAPHS")
    print("="*80)
    
    # Load results
    print("\n[1/3] Loading results...")
    results_data = load_all_results()
    
    # Print summary
    print("\n[2/3] Creating summary table...")
    create_summary_table(results_data)
    
    # Create graphs
    print("\n[3/3] Creating graphs...")
    heatmap_path = create_heatmap_graph(results_data)
    barchart_path = create_bar_chart(results_data)
    
    print("\n" + "="*80)
    print("✅ ALL GRAPHS CREATED!")
    print("="*80)
    print(f"Heatmap: {heatmap_path}")
    print(f"Bar Chart: {barchart_path}")
    print("="*80)

if __name__ == "__main__":
    main()

