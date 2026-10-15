with open('scripts/cubicasa_house.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'id") == "Door"' in l or 'class")' in l and 'Door' in l or '"Door"' in l:
        if 'if ' in l:
            print(f"Line {i+1}: {l.strip()}")
            for j in range(i, min(i+35, len(lines))):
                print(lines[j], end='')
            print('='*50)
