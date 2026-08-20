with open('test_eval.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('test_eval.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if 'get_ipython' not in line:
            f.write(line)
