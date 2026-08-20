import json

file_path = r"c:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\script-darts\ijf-solar-experiments.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "repo_path = r\"C:\\Users\\Anusha\\engression\\engression-ts\"" in source:
            new_source = [
                "import os\n",
                "import sys\n",
                "import shutil\n",
                "import subprocess\n",
                "\n",
                "# Clone the public repository if it doesn't exist\n",
                "repo_path = \"timeseries\"\n",
                "if not os.path.exists(repo_path):\n",
                "    subprocess.run([\"git\", \"clone\", \"https://github.com/pymaxx13/timeseries.git\"])\n",
                "\n",
                "# Add the repository to Python's import path\n",
                "repo_full_path = os.path.abspath(repo_path)\n",
                "if repo_full_path not in sys.path:\n",
                "    sys.path.insert(0, repo_full_path)\n",
                "\n",
                "print(\"Repository exists:\", os.path.exists(repo_full_path))\n",
                "print(\n",
                "    \"engressionts exists:\",\n",
                "    os.path.exists(os.path.join(repo_full_path, \"engressionts\"))\n",
                ")\n",
                "print(\"Repo path:\", repo_full_path)\n",
                "print(\"Python path contains repo:\", repo_full_path in sys.path)\n"
            ]
            cell["source"] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
