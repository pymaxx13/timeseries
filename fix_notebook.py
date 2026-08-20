import json
import os

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    if 'def get_fit_kwargs(' in source:
        new_source = '''# ============================================================
# CELL 17 - COVARIATE SUPPORT
# ============================================================

def get_fit_kwargs(
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
        cell['source'] = [line + '\n' for line in new_source.split('\n')][:-1]
        
    if 'def make_model(' in source:
        # replace NLinear model
        import re
        source = re.sub(
            r'(elif model_name == "NLinear":\s*return NLinearModel\(\s*)\*\*common',
            r'\1normalize=False,\n            **common',
            source
        )
        # Check if TCN replacement worked earlier
        if 'input_chunk_length=30' not in source and 'model_name == "TCN"' in source:
             source = re.sub(r'(model_name == "TCN":\s*return TCNModel\(\s*input_chunk_length=)24', r'\g<1>30', source)
        
        lines = []
        parts = source.split('\n')
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                lines.append(p + '\n')
            else:
                if p != '':
                    lines.append(p)
        cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixes applied successfully.")
