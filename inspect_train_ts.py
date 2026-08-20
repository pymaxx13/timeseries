import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\notebook8c0e95bba1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i in range(5, 20):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'train_ts' in source or 'train_y' in source:
            print(f"--- CELL {i} ---")
            print(source)
