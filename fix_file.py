import re

with open(r'd:\camoXpert_v2-claude-fix-modellevel-moe-param-0177FrkLFPTdKFtwVRC6C1px\trainers\optimized_trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace escaped sequences
if '\\n' in content or '\\"' in content:
    content = content.replace('\\n', '\n')
    content = content.replace('\\"', '"')
    content = content.replace('\\t', '    ')
    with open(r'd:\camoXpert_v2-claude-fix-modellevel-moe-param-0177FrkLFPTdKFtwVRC6C1px\trainers\optimized_trainer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed escaped characters')
else:
    print('No escaped characters found')
