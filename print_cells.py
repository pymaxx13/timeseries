import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r"c:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        print(f"--- Cell {i} ---")
        print(source)
        print()
