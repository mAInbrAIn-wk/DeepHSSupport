import sys

with open('src/simulation_v3.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('\\n', '\n')

with open('src/simulation_v3.py', 'w', encoding='utf-8') as f:
    f.write(code)
