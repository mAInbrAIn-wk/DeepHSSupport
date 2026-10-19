import re

with open('src/train_v3_multi_task.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_model = '''    model = Model(inputs=[seq_input, ctx_input], 
                  outputs=[out_grade_local, out_pass_local, out_status_global, out_grade_global])'''
                  
new_model = '''    model = Model(inputs={'exam_history': seq_input, 'next_exam_context': ctx_input}, 
                  outputs={
                      'out_grade_local': out_grade_local,
                      'out_pass_local': out_pass_local,
                      'out_status_global': out_status_global,
                      'out_grade_global': out_grade_global
                  })'''

content = content.replace(old_model, new_model)

with open('src/train_v3_multi_task.py', 'w', encoding='utf-8') as f:
    f.write(content)
