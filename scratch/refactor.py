import os

base_dir = r"c:\Users\Anusha\engression\engression-ts\engressionts\models"

# 1. Delete models
for file_name in ["endeepar.py", "endeepnpts.py", "envanillatransformer.py"]:
    file_path = os.path.join(base_dir, "neuralforecast", file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

# 2. Update neuralforecast/__init__.py
nf_init_path = os.path.join(base_dir, "neuralforecast", "__init__.py")
with open(nf_init_path, "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    if "endeepar" in line or "endeepnpts" in line or "envanillatransformer" in line:
        continue
    if "__all__" in line:
        line = line.replace(', "EnDeepAR"', '').replace(', "EnDeepNPTS"', '').replace(', "EnVanillaTransformer"', '')
    new_lines.append(line)
with open(nf_init_path, "w") as f:
    f.writelines(new_lines)

# 3. Update models/__init__.py
models_init_path = os.path.join(base_dir, "__init__.py")
with open(models_init_path, "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    if "EnVanillaTransformer" in line or "EnDeepAR" in line or "EnDeepNPTS" in line:
        if "from .neuralforecast import" in line:
            line = line.replace(", EnVanillaTransformer", "").replace(", EnDeepAR", "").replace(", EnDeepNPTS", "")
        else:
            continue
    new_lines.append(line)
with open(models_init_path, "w") as f:
    f.writelines(new_lines)

# 4. Rename darts models and update darts/__init__.py
darts_dir = os.path.join(base_dir, "darts")
rename_map = {
    "block_rnn_model.py": "enblock_rnn_model.py",
    "chronos2_model.py": "enchronos2_model.py",
    "dllinear_model.py": "endllinear_model.py",
    "nbeats.py": "enbeats.py",
    "nhits.py": "enhits.py",
    "nlinear_model.py": "ennlinear_model.py",
    "rnn_model.py": "enrnn_model.py",
    "tcn_model.py": "entcn_model.py",
    "tft_model.py": "entft_model.py",
    "tide_model.py": "entide_model.py",
    "transformer.py": "entransformer.py",
    "tsmixer_model.py": "entsmixer_model.py"
}

for old_name, new_name in rename_map.items():
    old_path = os.path.join(darts_dir, old_name)
    new_path = os.path.join(darts_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

darts_init_path = os.path.join(darts_dir, "__init__.py")
with open(darts_init_path, "r") as f:
    darts_init_content = f.read()

for old_name, new_name in rename_map.items():
    old_module = old_name[:-3]
    new_module = new_name[:-3]
    darts_init_content = darts_init_content.replace(f"engressionts.models.darts.{old_module}", f"engressionts.models.darts.{new_module}")

with open(darts_init_path, "w") as f:
    f.write(darts_init_content)
