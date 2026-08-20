import json
import os
import textwrap

notebook_path = r'C:\Users\Anusha\engression\engression-ts\engressionts\experiments\notebook8c0e95bba1.ipynb'

if not os.path.exists(notebook_path):
    print("Notebook not found at", notebook_path)
else:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Loaded notebook with {len(nb.get('cells', []))} cells.")
    
    for i, cell in enumerate(nb.get('cells', [])):
        print(f"\n{'='*40}\nCELL {i} [{cell['cell_type']}]\n{'='*40}")
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            print("--- SOURCE ---")
            print(textwrap.indent(source, '    ')[:1000])
            if len(source) > 1000: print("    ... (truncated)")
            
            outputs = cell.get('outputs', [])
            if outputs:
                print("\n--- OUTPUT ---")
                for out in outputs:
                    if out['output_type'] == 'stream':
                        text = "".join(out.get('text', []))
                        print(textwrap.indent(text, '    ')[-1000:]) # Last 1000 chars of stream
                    elif out['output_type'] == 'error':
                        print("    [ERROR]:", out.get('ename', ''), out.get('evalue', ''))
                        traceback = "\n".join(out.get('traceback', []))
                        print(textwrap.indent(traceback, '    '))
                    elif out['output_type'] in ['execute_result', 'display_data']:
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            print(textwrap.indent("".join(data['text/plain']), '    ')[:500])
                        else:
                            print(f"    [DATA of type: {list(data.keys())}]")
