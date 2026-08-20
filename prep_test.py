import json
import subprocess
import time

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Keep only the imports and the definition cells, and limit the MODELS to just NBEATS, and LIKELIHOODS to just Gaussian
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'MODELS =' in source:
            cell['source'] = ['MODELS = ["NBEATS"]\n', 'print("Models:", MODELS)\n']
        if 'LIKELIHOODS =' in source:
            cell['source'] = [
                'from darts.utils.likelihood_models import GaussianLikelihood\n',
                'LIKELIHOODS = {"Gaussian": GaussianLikelihood()}\n'
            ]
        if 'for model_name in MODELS:' in source and 'CSV_FILE =' in source:
            # We want to run it but only for 1 epoch for speed
            source = source.replace('N_EPOCHS', '1')
            
            lines = []
            parts = source.split('\n')
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    lines.append(p + '\n')
                else:
                    if p != '':
                        lines.append(p)
            cell['source'] = lines

test_notebook = 'test_eval.ipynb'
with open(test_notebook, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Running test notebook...")
