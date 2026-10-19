import re

with open('src/train_v3_multi_task.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_w_train = "w_train = {'out_grade_global': sample_weights_grade[tr_idx]}"
new_w_train = '''w_train = {
        'out_grade_local': np.ones(len(tr_idx)),
        'out_pass_local': np.ones(len(tr_idx)),
        'out_status_global': np.ones(len(tr_idx)),
        'out_grade_global': sample_weights_grade[tr_idx]
    }'''

old_w_test = "w_test = {'out_grade_global': sample_weights_grade[te_idx]}"
new_w_test = '''w_test = {
        'out_grade_local': np.ones(len(te_idx)),
        'out_pass_local': np.ones(len(te_idx)),
        'out_status_global': np.ones(len(te_idx)),
        'out_grade_global': sample_weights_grade[te_idx]
    }'''

content = content.replace(old_w_train, new_w_train)
content = content.replace(old_w_test, new_w_test)

with open('src/train_v3_multi_task.py', 'w', encoding='utf-8') as f:
    f.write(content)
