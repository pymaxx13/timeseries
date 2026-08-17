import os

nf_dir = r"c:\Users\Anusha\engression\engression-ts\engressionts\models\neuralforecast"

for fn in os.listdir(nf_dir):
    if fn == "__init__.py" or fn == "enhint.py" or not fn.endswith(".py"):
        continue
        
    fp = os.path.join(nf_dir, fn)
    print(f"Processing {fn}...")
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace num_samples in signature
    if "num_samples: int = 20," in content:
        content = content.replace(
            "num_samples: int = 20,",
            "num_samples_train: int = 20, num_samples=None,"
        )
    else:
        print(f"WARNING: num_samples: int = 20, not found in {fn}")
        
    # Replace num_samples in super().__init__ call
    if "num_samples=num_samples," in content:
        content = content.replace(
            "num_samples=num_samples,",
            "num_samples_train=num_samples_train, num_samples=num_samples,"
        )
    else:
        # Check for different spacing
        if "num_samples = num_samples," in content:
            content = content.replace(
                "num_samples = num_samples,",
                "num_samples_train=num_samples_train, num_samples=num_samples,"
            )
        else:
            print(f"WARNING: num_samples=num_samples, not found in {fn}")
            
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)

print("Modification complete!")
