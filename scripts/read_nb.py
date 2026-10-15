with open(r'C:\Users\ayebk\.gemini\antigravity\brain\387c7b87-863d-4c00-831b-93abdc64d4d4\.system_generated\steps\2598\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('## Parsed labels')
while pos != -1:
    tag = '"source": ['
    code_pos = text.find(tag, pos)
    if code_pos == -1:
        break
    end_pos = text.find(']', code_pos)
    print("--- FOUND SOURCE ---")
    print(text[code_pos:end_pos+1])
    pos = text.find('cell_type', end_pos)
