
import ast
import json
import os

file_path = 'output/ep001_sample_transcription.json'

# Read the raw content
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# The content is in a format that resembles a Python dictionary with triple-quoted strings.
# We can use ast.literal_eval to safely parse this python-syntax string into a dict.
try:
    data = ast.literal_eval(content)
    print("Successfully parsed the file as a Python dictionary.")
    
    # Now write it back as valid JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Successfully converted {file_path} to valid JSON.")

except Exception as e:
    print(f"Error parsing file: {e}")
    # Fallback to manual fix if ast fails (e.g. if true/false are lowercase json style instead of Python True/False)
    # JSON uses 'true', 'false', 'null'. Python uses 'True', 'False', 'None'.
    # If the file uses JSON booleans, ast.literal_eval might fail unless we replace them.
    # Let's inspect the `view_file` output again.
    # Line 5: "duration_seconds": 120.5,
    # No booleans visible in the first view.
    # But wait, step2 output suggests "Dry Run: False" which is a python print.
    # The JSON file itself:
    # "segments": [ ... ]
    # The segments have "start", "end", "text".
    # I don't see any true/false/null literals in the file shown.
    # Be safe and replace json literals with python literals before literal_eval just in case.
    
    import re
    content_fixed = re.sub(r'\btrue\b', 'True', content)
    content_fixed = re.sub(r'\bfalse\b', 'False', content_fixed)
    content_fixed = re.sub(r'\bnull\b', 'None', content_fixed)
    try:
        data = ast.literal_eval(content_fixed)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully converted {file_path} to valid JSON (after literal fix).")
    except Exception as e2:
         print(f"Error parsing file after literal fix: {e2}")
