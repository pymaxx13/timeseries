import json
import re
import os

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # 1. Update QUANTILES
    if 'QUANTILES = [' in source:
        source = re.sub(r'QUANTILES\s*=\s*\[.*?\]', 'QUANTILES = [\n    0.1,\n    0.5,\n    0.9\n]', source, flags=re.DOTALL)
        
    # 2. Update TCN Input Chunk Length
    if 'model_name == "TCN":' in source:
        source = re.sub(r'(model_name == "TCN":\s*return TCNModel\(\s*input_chunk_length=)24', r'\g<1>30', source)
        
    # 3. Update get_fit_kwargs
    if 'def get_fit_kwargs(' in source:
        # replace the signature and body
        new_func = '''def get_fit_kwargs(
    model,
    target,
    past_covs=None
):

    kwargs = {
        "series": target,
        "verbose": False,
        "dataloader_kwargs": {"num_workers": 0},
    }

    if getattr(
        model,
        "supports_past_covariates",
        False
    ):
        if past_covs is not None:
            kwargs["past_covariates"] = past_covs
        else:
            kwargs["past_covariates"] = base_train_pc

    return kwargs
'''
        # we will just replace the whole cell source assuming it only contains this function and some comments
        # Let's be safer and replace using regex
        source = re.sub(r'def get_fit_kwargs\(.*?:.*?return kwargs\n', new_func, source, flags=re.DOTALL)
        
    # 4. Update CELL 18 (evaluate_base_model) and CELL 22 (evaluate_conformal_model) for clipping
    if 'fc = y_scaler.inverse_transform(' in source:
        if 'np.clip' not in source:
            source = re.sub(
                r'(fc = y_scaler\.inverse_transform\(\s*fc_sc\s*\)\n)',
                r'\1\n        fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))\n',
                source
            )
            
    # 5. Update Likelihood training data in CELL 19
    if 'fit_kwargs = (' in source and 'base_train_y_sc' in source and 'get_fit_kwargs' in source and 'for model_name in MODELS:' in source:
        source = source.replace(
            '''            fit_kwargs = (
                get_fit_kwargs(
                    model,
                    base_train_y_sc
                )
            )''',
            '''            fit_kwargs = (
                get_fit_kwargs(
                    model,
                    train_y_sc,
                    past_covs=train_pc
                )
            )'''
        )

    # 6. Update Conformal naive model evaluate cell (CELL 22 or similar) 
    # Actually step 4 already applies to any cell with c = y_scaler.inverse_transform(
    # Let's double check Conformal eval if it uses predict and inverse_transform
    # Conformal naive uses cp_model.predict, so step 4 handles it. Wait, does ConformalQRModel do that too? Yes.

    # split the source back into lines for the cell
    if "\n" in source:
        lines = [line + '\n' for line in source.split('\n')]
        # remove trailing newline from last element if the original source didn't end with one
        if source and not source.endswith('\n'):
            lines[-1] = lines[-1].rstrip('\n')
        else:
            # wait, if source ended with \n, split('\n') gives empty string at end, which becomes '\n'
            # Let's use a simpler approach:
            lines = []
            parts = source.split('\n')
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    lines.append(p + '\n')
                else:
                    if p != '':
                        lines.append(p)
    else:
        lines = [source]
        
    cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Modifications successfully applied to the notebook.")
