from concurrent.futures import ThreadPoolExecutor
from bing_image_downloader import downloader
from serpapi import GoogleSearch
import requests
import os
from PIL import Image
import imagehash

# === CONFIGURACIÓN ===
SEARCH_QUERIES = [
    'chrysler prowler',
    'plymouth prowler',
    'chrysler prowler side view',
    'chrysler prowler front view',
    'chrysler prowler rear view',
    'chrysler prowler modified',
    'chrysler prowler parked',
    'chrysler prowler at car show',
    'chrysler prowler in urban setting',
    'chrysler prowler on highway',
    'chrysler prowler on race track',
    'chrysler prowler top view',
    'chrysler prowler aerial view',
    'chrysler prowler side profile',
    'chrysler prowler front angle low shot',
    'chrysler prowler sunset drive',
    'chrysler prowler in mountain road',
    'chrysler prowler at night'
]





BING_LIMIT = 400  # Número de imágenes por búsqueda en Bing
GOOGLE_LIMIT = 400  # Número de imágenes por búsqueda en Google
OUTPUT_DIR = 'assets/scraping/prowler'  # Carpeta donde se guardarán las imágenes
GOOGLE_API_KEY = "78e4c4ae8ce47dc647a1fdbfdf3254d325eac837e279ab616826afaf4fcbd0c9"

# === FUNCIONES ===
def download_images_from_bing(query):
    """Descarga imágenes desde Bing."""
    print(f"Descargando imágenes desde Bing para: {query}")
    downloader.download(query, limit=BING_LIMIT, output_dir=OUTPUT_DIR, adult_filter_off=True, force_replace=False, timeout=60, verbose=False)

def download_images_from_google(query):
    """Descarga imágenes desde Google usando SerpAPI."""
    print(f"Descargando imágenes desde Google para: {query}")
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "ijn": "0",
        "api_key": GOOGLE_API_KEY
    })
    results = search.get_dict()
    image_urls = [result['original'] for result in results.get('images_results', [])[:GOOGLE_LIMIT]]

    query_dir = os.path.join(OUTPUT_DIR, query.replace(' ', '_'))
    os.makedirs(query_dir, exist_ok=True)

    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(query_dir, f"google_{i}.jpg"), 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error descargando {url}: {e}")

def remove_duplicates(folder):
    """Elimina imágenes duplicadas en una carpeta usando hashes."""
    print("Eliminando imágenes duplicadas...")
    seen_hashes = set()
    for root, _, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                img_hash = imagehash.average_hash(Image.open(filepath))
                if img_hash in seen_hashes:
                    os.remove(filepath)
                else:
                    seen_hashes.add(img_hash)
            except Exception as e:
                print(f"Error con {filename}: {e}")

# === MAIN ===
if __name__ == "__main__":
    # Paralelizar descargas desde Bing y Google
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Bing descargas
        executor.map(download_images_from_bing, SEARCH_QUERIES)

        # Google descargas
        executor.map(download_images_from_google, SEARCH_QUERIES)

    # Eliminar duplicados
    remove_duplicates(OUTPUT_DIR)

    print("Descarga completa y dataset limpio.")
