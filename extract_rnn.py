import json
def extract(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'EnBlockRNNModel' in source and 'n_epochs' in source:
                print(source)
extract(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb')
