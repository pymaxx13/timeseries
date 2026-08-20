import json

file_path = r"c:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "project_root = Path.cwd().parents[2]" in source:
            new_source = [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "try:\n",
                "    project_root = Path.cwd().parents[2]   # .../engression-ts\n",
                "    sys.path.insert(0, str(project_root))\n",
                "except IndexError:\n",
                "    # On Kaggle, the notebook is run in the root directory (e.g., /kaggle/working)\n",
                "    # so Path.cwd().parents[2] will fail. \n",
                "    # We already added the cloned repo to sys.path in an earlier cell.\n",
                "    pass\n"
            ]
            cell["source"] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
