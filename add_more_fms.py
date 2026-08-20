import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Add FMs to MODELS
        if 'MODELS =' in source and 'PatchTSTFM' not in source:
            source = source.replace('    "TimesFM2p5",\n]', '    "TimesFM2p5",\n    "PatchTSTFM",\n    "TiREx",\n]')
            
        if 'def make_model' in source and 'PatchTSTFM' not in source:
            new_logic = '''
    elif model_name == "PatchTSTFM":
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("patchtst", "engressionts/models/darts-original/patchtst_fm_model.py")
        patchtst = importlib.util.module_from_spec(spec)
        sys.modules["patchtst"] = patchtst
        spec.loader.exec_module(patchtst)
        return patchtst.PatchTSTFMModel(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            n_epochs=N_EPOCHS,
        )

    elif model_name == "TiREx":
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("tirex", "engressionts/models/darts-original/tirex_model.py")
        tirex = importlib.util.module_from_spec(spec)
        sys.modules["tirex"] = tirex
        spec.loader.exec_module(tirex)
        return tirex.TiRExModel(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            n_epochs=N_EPOCHS,
            accept_license=True,
        )
'''
            source = source.replace('    else:\n        raise ValueError(\n            f"Unknown model: {model_name}"\n        )', new_logic + '\n    else:\n        raise ValueError(\n            f"Unknown model: {model_name}"\n        )')
            source = source.replace('    else:\n        raise ValueError(f"Unknown model: {model_name}")', new_logic + '\n    else:\n        raise ValueError(f"Unknown model: {model_name}")')

        if 'for likelihood_name in LIKELIHOODS:' in source:
            source = source.replace('["Chronos2", "TimesFM2p5"]', '["Chronos2", "TimesFM2p5", "PatchTSTFM", "TiREx"]')

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

print("Added PatchTSTFM and TiREx")
