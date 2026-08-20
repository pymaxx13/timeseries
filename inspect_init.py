import re

def extract_init(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find class definition and its __init__
    for match in re.finditer(r'class \w+Model\(FoundationModel\):.*?def __init__\((.*?)\):', content, re.DOTALL):
        return match.group(1)
    
    # Fallback to any class ending in Model
    for match in re.finditer(r'class \w+Model\(.*?\):.*?def __init__\((.*?)\):', content, re.DOTALL):
        return match.group(1)
    
    return "Not found"

print("PatchTSTFMModel init:")
print(extract_init(r'C:\Users\Anusha\engression\engression-ts\engressionts\models\darts-original\patchtst_fm_model.py'))
print("\nTiRExModel init:")
print(extract_init(r'C:\Users\Anusha\engression\engression-ts\engressionts\models\darts-original\tirex_model.py'))
