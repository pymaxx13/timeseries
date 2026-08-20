import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def evaluate_conformal_model' in source:
            print(source)
