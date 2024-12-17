from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Carga del dataset
data = pd.read_csv("ruta_al_dataset.csv")
documents = data['texto'].tolist()

# Vectorización
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo ligero para embeddings
embeddings = model.encode(documents)

# Crear el índice
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
