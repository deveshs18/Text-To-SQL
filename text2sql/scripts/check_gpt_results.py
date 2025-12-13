"""Check why GPT-4o-mini has same numbers across techniques."""
import pandas as pd
from pathlib import Path

results_dir = Path('text2sql/scripts/data/results')
techniques = ['Few-Shot', 'CoT', 'LtM', 'EG']

print('GPT-4o-mini EX with RAG counts:')
print('='*60)
for t in techniques:
    df = pd.read_csv(results_dir / f'openai_gpt-4o-mini_{t}_results.csv')
    df_filtered = df[df['With_RAG_Success'] != 'SKIPPED']
    passed = sum(df_filtered['With_RAG_EX'] == '✅')
    total = len(df_filtered)
    print(f'{t:15} {passed}/{total} ({passed/total*100:.1f}%)')


