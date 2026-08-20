import json

def extract_cells(filepath, keyword):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if keyword in source:
                return source
    return None

print("=== BASELINES: EXPERIMENT CONFIG ===")
print(extract_cells(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb', 'DATASET_NAME ='))

print("\n=== ENGRESSION: EXPERIMENT CONFIG ===")
print(extract_cells(r'C:\Users\Anusha\engression\engression-ts\ijf-solar-experiments.ipynb', 'DATASET_NAME ='))

print("\n=== BASELINES: LIKELIHOODS ===")
print(extract_cells(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb', 'LIKELIHOODS ='))

