import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'NegativeBinomialLikelihood' in source:
            source = source.replace(',\n    NegativeBinomialLikelihood,\n', '\n')
            source = source.replace(', NegativeBinomialLikelihood', '')
            
            final_lines = []
            parts = source.split('\n')
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    final_lines.append(p + '\n')
                else:
                    if p != '':
                        final_lines.append(p)
            cell['source'] = final_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cleaned dangling imports.")
