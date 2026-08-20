import json
import re

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '"n_epochs": 0' in source:
            source = source.replace('"n_epochs": 0', '"n_epochs": 30')
            source = source.replace('n_epochs=0 to avoid OOM', 'n_epochs=30')
            source = source.replace('like n_epochs=0', 'like n_epochs=30')
            
            lines = []
            parts = source.split('\n')
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    lines.append(p + '\n')
                else:
                    if p != '':
                        lines.append(p)
            cell['source'] = lines
            changed = True

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Updated n_epochs to 30 in ijf-solar-experiments.ipynb")
else:
    print("Could not find n_epochs: 0")
