import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\notebook8c0e95bba1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'train_y_sc' in source and '.shape' in source:
            print("Found shape check")
