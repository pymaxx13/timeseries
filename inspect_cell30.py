import json
notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\notebook8c0e95bba1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'for idx, (model_name, model_class, extra_kwargs) in enumerate(MODELS_TO_RUN, 1):' in source:
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out['output_type'] == 'stream':
                    print(out.get('text', [])[-20:])
                elif out['output_type'] == 'error':
                    print("ERROR:", out.get('ename'), out.get('evalue'))
