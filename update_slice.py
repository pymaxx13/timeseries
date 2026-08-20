import json
import re

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '.slice_end(forecast_start)' in source:
            source = source.replace('.slice_end(forecast_start)', '.drop_after(forecast_start, keep_point=False)')
            
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

print("Updated slice_end to drop_after.")
