import os
import ast

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    
    results = []
    for cls in classes:
        # Find __init__ method
        init_method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
        has_likelihood = False
        likelihood_annotation = None
        
        if init_method:
            for arg in init_method.args.args + init_method.args.kwonlyargs:
                if arg.arg == 'likelihood':
                    has_likelihood = True
                    if arg.annotation:
                        likelihood_annotation = ast.unparse(arg.annotation)
        
        base_classes = [ast.unparse(b) for b in cls.bases]
        
        results.append({
            'class': cls.name,
            'bases': base_classes,
            'has_likelihood': has_likelihood,
            'likelihood_annotation': likelihood_annotation,
        })
    return results

def main():
    original_dir = 'c:/Users/Anusha/engression/engression-ts/engressionts/models/darts-original'
    wrapper_dir = 'c:/Users/Anusha/engression/engression-ts/engressionts/models/darts'
    
    print("=== ORIGINAL MODELS ===")
    for filename in os.listdir(original_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(original_dir, filename)
            res = analyze_file(filepath)
            for r in res:
                if 'Model' in r['class'] or r['class'] in ['NBEATSModel', 'NHITSModel', 'NLinearModel', 'DLinearModel']:
                    print(f"{filename} -> {r['class']}({', '.join(r['bases'])}):")
                    print(f"  likelihood: {r['has_likelihood']} ({r['likelihood_annotation']})")
                    
    print("\n=== WRAPPER MODELS ===")
    for filename in os.listdir(wrapper_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(wrapper_dir, filename)
            res = analyze_file(filepath)
            for r in res:
                if r['class'].startswith('En'):
                    print(f"{filename} -> {r['class']}({', '.join(r['bases'])}):")
                    print(f"  likelihood: {r['has_likelihood']} ({r['likelihood_annotation']})")

if __name__ == '__main__':
    main()
