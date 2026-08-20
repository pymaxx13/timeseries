import importlib.util
import sys

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

try:
    tirex = load_module("tirex_model", r"C:\Users\Anusha\engression\engression-ts\engressionts\models\darts-original\tirex_model.py")
    print(dir(tirex))
except Exception as e:
    print(f"Error: {e}")
