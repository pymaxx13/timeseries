import json

def extract_cells(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    results = {}
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'DATASET_NAME =' in source:
                results['CONFIG'] = source
            if 'test_windows =' in source or 'get_test_windows' in source:
                results['TEST_WINDOWS'] = source
            if 'def make_model' in source or 'def get_model' in source:
                results['MAKE_MODEL'] = source
            if 'evaluate_' in source:
                if 'EVAL_FUNC' not in results:
                    results['EVAL_FUNC'] = ""
                results['EVAL_FUNC'] += source + "\n\n"
            if 'def get_fit_kwargs' in source:
                results['FIT_KWARGS'] = source
    return results

base = extract_cells(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb')
engr = extract_cells(r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb')

for key in base:
    print(f"\n=== BASELINES: {key} ===")
    print(base[key][:1000])

for key in engr:
    print(f"\n=== ENGRESSION: {key} ===")
    print(engr[key][:1000])

