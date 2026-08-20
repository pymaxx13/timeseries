import json

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Remove NegativeBinomialLikelihood import
        if 'NegativeBinomialLikelihood' in source:
            source = source.replace('from darts.utils.likelihood_models import (\n    GaussianLikelihood,\n    QuantileRegression,\n    NegativeBinomialLikelihood\n)', 'from darts.utils.likelihood_models import (\n    GaussianLikelihood,\n    QuantileRegression\n)')
            source = source.replace('from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression, NegativeBinomialLikelihood', 'from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression')
            
            # Remove from LIKELIHOODS dictionary
            # It might look like: "NegativeBinomial": NegativeBinomialLikelihood(),
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if '"NegativeBinomial"' not in line and "'NegativeBinomial'" not in line:
                    new_lines.append(line)
                else:
                    # If there's a trailing comma on the previous line we might need to handle it, 
                    # but usually JSON/Python is fine with trailing commas.
                    pass
            source = '\n'.join(new_lines)
            
            # Update the cell source format
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

print("Removed NegativeBinomialLikelihood from notebook.")
