import json
import re
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Path Fix
        if 'engressionts/models/darts-original/patchtst_fm_model.py' in source:
            source = source.replace('engressionts/models/darts-original/patchtst_fm_model.py', '../../models/darts-original/patchtst_fm_model.py')
        
        if 'engressionts/models/darts-original/tirex_model.py' in source:
            source = source.replace('engressionts/models/darts-original/tirex_model.py', '../../models/darts-original/tirex_model.py')
            
        # 2. ConformalNaive Quantiles Fix
        if 'quantiles=[' in source and '0.025,' in source and 'ConformalNaiveModel(' in source:
            # We want to replace quantiles=[...], with quantiles=QUANTILES,
            source = re.sub(r'quantiles=\[\s*0\.025,\s*0\.975\s*\],', 'quantiles=QUANTILES,', source)
            
        lines = []
        parts = source.split('\n')
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                lines.append(p + '\n')
            else:
                if p != '':
                    lines.append(p)
        cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Applied notebook fixes.")
