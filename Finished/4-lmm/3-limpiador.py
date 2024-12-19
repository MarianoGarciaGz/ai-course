import re
import pdfplumber
import os
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

text_dir = "./assets/web"
clean_dir = "./limpios"
os.makedirs(clean_dir, exist_ok=True)

for text_file in os.listdir(text_dir):
    if text_file.endswith(".txt"):
        text_path = os.path.join(text_dir, text_file)
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)
        
        output_path = os.path.join(clean_dir, text_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Texto limpio guardado en: {output_path}")
