import os
import json

# Ruta a la carpeta con los archivos .txt
txt_folder = "/content/txt_files"

# Crear una lista para almacenar los datos procesados
dataset = []

# Leer todos los archivos .txt
for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(txt_folder, filename), "r", encoding="utf-8") as f:
            content = f.read()
            dataset.append({"prompt": f"Contenido de {filename}", "response": content})

# Guardar como JSONL
with open("/content/dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

