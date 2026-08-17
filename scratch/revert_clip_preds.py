import os

files_to_clean = [
    r"engressionts/models/darts/enpatchtst_fm_model.py",
    r"engressionts/models/darts/entimesfm2p5_model.py",
    r"engressionts/models/darts/entirex_model.py"
]

for fp in files_to_clean:
    if not os.path.exists(fp):
        print(f"Skipping {fp}")
        continue
        
    print(f"Cleaning {fp}...")
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if "clip_preds" in line:
            print(f"  Removed: {line.strip()}")
            continue
        new_lines.append(line)
        
    with open(fp, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print("Revert complete!")
