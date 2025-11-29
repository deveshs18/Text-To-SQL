"""
Comprehensive analysis of RAG vs No-RAG impact on different prompting techniques.
Reads all evaluation results and provides detailed comparison.
"""
import pandas as pd
import glob
from pathlib import Path

def parse_latency(latency_str):
    """Parse latency string to float."""
    if isinstance(latency_str, str):
        return float(latency_str.replace('s', '').strip())
    return float(latency_str)

def analyze_rag_impact():
    """Analyze RAG impact across all techniques and models."""
    results_dir = Path(__file__).parent / "data" / "results"
    
    # Read all result files
    result_files = list(results_dir.glob("*_results.csv"))
    
    print("="*80)
    print("COMPREHENSIVE RAG vs NO-RAG ANALYSIS")
    print("="*80)
    print("\nReading evaluation results...")
    
    all_results = {}
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            
            # Determine model
            if 'ollama' in file.name.lower():
                model = 'Ollama'
            elif 'gpt' in file.name.lower() or 'openai' in file.name.lower():
                model = 'GPT-4o-mini'
            else:
                continue
            
            # Extract technique
            technique = None
            for tech in ['Few-Shot', 'CoT', 'LtM', 'EG']:
                if tech in file.name:
                    technique = tech
                    break
            
            if not technique:
                continue
            
            key = f"{model}_{technique}"
            all_results[key] = df
            
            print(f"  Loaded: {model} - {technique} ({len(df)} cases)")
            
        except Exception as e:
            print(f"  Error reading {file.name}: {e}")
            continue
    
    print(f"\nTotal result sets loaded: {len(all_results)}")
    print("="*80)
    
    # Analyze each model-technique combination
    analysis_results = []
    
    for key, df in all_results.items():
        model, technique = key.split('_', 1)
        
        # Calculate metrics
        total = len(df)
        
        # Success rates
        with_rag_success = sum(df['With_RAG_Success'] == '✅')
        without_rag_success = sum(df['Without_RAG_Success'] == '✅')
        success_improvement = ((with_rag_success - without_rag_success) / total) * 100
        
        # Execution Accuracy (EX)
        with_rag_ex = sum(df['With_RAG_EX'] == '✅')
        without_rag_ex = sum(df['Without_RAG_EX'] == '✅')
        ex_improvement = ((with_rag_ex - without_rag_ex) / total) * 100
        
        # Semantic Match (SM)
        with_rag_sm = sum(df['With_RAG_SM'] == '✅')
        without_rag_sm = sum(df['Without_RAG_SM'] == '✅')
        sm_improvement = ((with_rag_sm - without_rag_sm) / total) * 100
        
        # F1-Score
        with_rag_f1 = pd.to_numeric(df['With_RAG_F1'], errors='coerce').mean()
        without_rag_f1 = pd.to_numeric(df['Without_RAG_F1'], errors='coerce').mean()
        f1_improvement = with_rag_f1 - without_rag_f1
        
        # RAG helped cases
        rag_helped = sum(df['RAG_Helped'] == '✅')
        rag_hurt = sum(df['RAG_Helped'] == '❌')
        rag_neutral = sum(df['RAG_Helped'] == '➖')
        
        analysis_results.append({
            'Model': model,
            'Technique': technique,
            'Total_Cases': total,
            'With_RAG_Success': with_rag_success,
            'Without_RAG_Success': without_rag_success,
            'Success_Improvement_%': success_improvement,
            'With_RAG_EX': with_rag_ex,
            'Without_RAG_EX': without_rag_ex,
            'EX_Improvement_%': ex_improvement,
            'With_RAG_SM': with_rag_sm,
            'Without_RAG_SM': without_rag_sm,
            'SM_Improvement_%': sm_improvement,
            'With_RAG_F1': with_rag_f1,
            'Without_RAG_F1': without_rag_f1,
            'F1_Improvement': f1_improvement,
            'RAG_Helped': rag_helped,
            'RAG_Hurt': rag_hurt,
            'RAG_Neutral': rag_neutral
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    
    # Save detailed analysis
    output_file = results_dir / "rag_impact_analysis.csv"
    analysis_df.to_csv(output_file, index=False)
    print(f"\n[OK] Detailed analysis saved to: {output_file}")
    
    return analysis_df

if __name__ == "__main__":
    df = analyze_rag_impact()
    print("\nAnalysis complete!")


