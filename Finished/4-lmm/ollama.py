from datasets import load_dataset

# Cargar el archivo JSONL como un dataset
dataset = load_dataset("json", data_files="./dataset.jsonl")

# Verificar el contenido
print(dataset)
print(dataset["train"][0])  # Muestra el primer ejemplo
