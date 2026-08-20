import json
def extract(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'epochs' in source or 'BATCH_SIZE' in source or 'SEED' in source or 'n_epochs' in source:
                print("== MATCH ==")
                print(source[:500])
extract(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb')
