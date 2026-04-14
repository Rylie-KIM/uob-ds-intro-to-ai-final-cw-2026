from pathlib import Path
p = Path(r'C:\Masters\Text Analytics\AI_Coursework')
print(p.exists())          # does the parent exist?
print(list(p.iterdir()))  