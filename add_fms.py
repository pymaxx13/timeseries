import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Add FM imports
        if 'TSMixerModel,' in source and 'Chronos2Model' not in source:
            source = source.replace('TSMixerModel,\n)', 'TSMixerModel,\n    Chronos2Model,\n    TimesFM2p5Model,\n)')
            
        # Add FMs to MODELS
        if 'MODELS =' in source and 'Chronos2' not in source:
            source = source.replace('    "TSMixer",\n    "TFT",\n]', '    "TSMixer",\n    "TFT",\n    "Chronos2",\n    "TimesFM2p5",\n]')
            
        # Update make_model
        if 'def make_model' in source and 'Chronos2Model' not in source:
            fm_logic = """    elif model_name == "Chronos2":
        return Chronos2Model(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=RANDOM_STATE,
        )
    elif model_name == "TimesFM2p5":
        return TimesFM2p5Model(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=RANDOM_STATE,
        )"""
            source = source.replace('    else:\n        raise ValueError(f"Unknown model: {model_name}")', fm_logic + '\n    else:\n        raise ValueError(f"Unknown model: {model_name}")')
            
        # Update evaluation loop to skip incompatible likelihoods for FM
        if 'for likelihood_name in LIKELIHOODS:' in source and 'Foundation models only support Quantile' not in source:
            skip_logic = """
        # Foundation models only support Quantile in this setup
        if model_name in ["Chronos2", "TimesFM2p5"] and likelihood_name != "Quantile":
            print(f"Skipping {likelihood_name} for {model_name} (Zero-shot Foundation Models default to Quantile)")
            continue
"""
            source = source.replace('for likelihood_name in LIKELIHOODS:', 'for likelihood_name in LIKELIHOODS:' + skip_logic)

        # Update cell source
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

print("Added Foundation Models.")
