import pandas as pd

# Ruta al archivo JSON
json_file = "textos_procesados.json"  # Ruta relativa o absoluta del archivo JSON

# Cargar datos del JSON
data = pd.read_json(json_file)

# Ruta para guardar el archivo CSV (usa una ruta válida)
csv_file = "textos_procesados.csv"  # Guarda en el directorio actual
# csv_file = r"C:\Users\Oscar Fuentes\Documents\textos_procesados.csv"  # Ejemplo de ruta absoluta

# Convertir a CSV
data.to_csv(csv_file, index=False, encoding='utf-8')

print(f"Archivo convertido y guardado en: {csv_file}")
