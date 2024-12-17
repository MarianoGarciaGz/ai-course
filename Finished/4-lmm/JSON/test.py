from datasets import load_from_disk
dataset = load_from_disk("mi_dataset")
print(dataset)
# Ver las primeras 3 filas del dataset
print(dataset[:3])

