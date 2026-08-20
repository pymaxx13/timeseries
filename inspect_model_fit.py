import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\notebook8c0e95bba1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i in range(20, min(30, len(nb['cells']))):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        print(f"\n--- CELL {i} ---")
        print(source)
