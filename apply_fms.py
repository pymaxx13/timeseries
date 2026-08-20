import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if 'def make_model' in source and 'Chronos2' not in source:
            old_str = '    else:\n        raise ValueError(\n            f"Unknown model: {model_name}"\n        )'
            new_str = '''    elif model_name == "Chronos2":
        return Chronos2Model(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=SEED,
            n_epochs=N_EPOCHS,
        )

    elif model_name == "TimesFM2p5":
        return TimesFM2p5Model(
            input_chunk_length=24,
            output_chunk_length=PRED_LEN,
            likelihood=likelihood,
            random_state=SEED,
            n_epochs=N_EPOCHS,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )'''
            if old_str in source:
                source = source.replace(old_str, new_str)
            else:
                # If exact spacing is different
                import re
                source = re.sub(r'    else:\s+raise ValueError\(\s+f"Unknown model: \{model_name\}"\s+\)', new_str, source)

        if 'for likelihood_name in LIKELIHOODS:' in source and 'Foundation models only support' not in source:
            source = source.replace('for likelihood_name in LIKELIHOODS:', 'for likelihood_name in LIKELIHOODS:\n\n        if model_name in ["Chronos2", "TimesFM2p5"] and likelihood_name != "Quantile":\n            print(f"Skipping {likelihood_name} for {model_name}")\n            continue')

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

print("Updates applied.")
