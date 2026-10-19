import re

with open('src/run_v4_universes.py', 'r', encoding='utf-8') as f:
    content = f.read()

bad_snippet = '''        # 2. Studierende klonen, damit ihr initialer Zustand in jedem Universum gleich ist
        import copy
        studierende_klohn = copy.deepcopy(base_studierende)'''
        
good_snippet = '''        # 2. Studierende klonen, damit ihr initialer Zustand in jedem Universum gleich ist
        studierende_klohn = copy.deepcopy(base_studierende)'''

content = content.replace(bad_snippet, good_snippet)

with open('src/run_v4_universes.py', 'w', encoding='utf-8') as f:
    f.write(content)
