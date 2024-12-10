import os
import random
import csv

import pygame
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from PIL import Image, ImageFilter, ImageEnhance


# ------------------------ Inicialización de Pygame ------------------------
pygame.init()

# Dimensiones de la pantalla
WIDTH, HEIGHT = 800, 400
pantalla = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Fuente y colores
FUENTE = pygame.font.Font(None, 36)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# ------------------------ Carga de recursos (Imágenes) ------------------------
jugador_frames = [
    pygame.image.load('./assets/sprites/mono_frame_1.png'),
    pygame.image.load('./assets/sprites/mono_frame_1.png')
]
bala_img = pygame.image.load('./assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('./assets/game/fondo.png')
nave_img = pygame.image.load('./assets/game/ufo.png')
menu_img = pygame.image.load('./assets/game/menu.png')

# Escalar el fondo al tamaño de la ventana
fondo_img = pygame.transform.scale(fondo_img, (WIDTH, HEIGHT))

# ------------------------ Variables globales del juego ------------------------
jugador = pygame.Rect(50, HEIGHT - 100, 32, 48)
bala = pygame.Rect(WIDTH - 50, HEIGHT - 90, 16, 16)
nave = pygame.Rect(WIDTH - 100, HEIGHT - 100, 64, 64)

# Estado del jugador (Salto y gravedad)
salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

# Estado de la bala
velocidad_bala = -10
bala_disparada = False

# Posiciones del fondo para efecto "scroll"
fondo_x1 = 0
fondo_x2 = WIDTH

# Estado del juego
pausa = False
menu_activo = True
modo_auto = False

# Datos para entrenamiento
datos_para_csv = []


# ------------------------ Clase para manejo de modelos (Machine Learning) ------------------------
class ModelManager:
    def __init__(self):
        self.arbol = DecisionTreeClassifier()
        self.red_neuronal = None
        self.entrenado = False

    def cargar_modelo_h5(self, filepath="modelo_red.h5"):
        # Carga un modelo de red neuronal previamente entrenado (Keras)
        try:
            self.red_neuronal = keras.models.load_model(filepath)
            self.red_neuronal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.entrenado = True
            print(f"Modelo de red neuronal cargado desde {filepath}.")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}. Por favor, entrena y guarda un modelo primero.")
            self.entrenado = False

    def predecir_red(self, velocidad, distancia, umbral=0.5):
        # Realiza una predicción con la red neuronal dada una velocidad y distancia
        if not self.entrenado or self.red_neuronal is None:
            raise ValueError("El modelo de red neuronal no está entrenado.")
        entrada = np.array([[velocidad, distancia]])
        prediccion = self.red_neuronal.predict(entrada)[0][0]
        return 1 if prediccion >= umbral else 0

    def entrenar_arbol(self, datos):
        # Entrena el árbol de decisiones con datos en memoria (numpy array)
        self.arbol = DecisionTreeClassifier()
        self.entrenado = False
        if len(datos) > 0:
            X, y = datos[:, :2], datos[:, 2]
            self.arbol.fit(X, y)
            self.entrenado = True
            print("Árbol entrenado correctamente.")
        else:
            print("No hay datos para entrenar.")

    def predecir_arbol(self, velocidad, distancia):
        # Predice usando el árbol de decisiones ya entrenado
        if not self.entrenado:
            raise ValueError("El modelo no está entrenado.")
        return int(self.arbol.predict([[velocidad, distancia]])[0])

    def guardar_modelo(self, filepath="modelo_arbol.pkl"):
        # Guarda el árbol de decisiones entrenado
        if self.entrenado:
            joblib.dump(self.arbol, filepath)
            print(f"Modelo guardado en {filepath}.")
        else:
            print("El modelo no está entrenado, no se guardará.")

    def entrenar_arbol_desde_csv(self, filepath="dataset.csv"):
        # Entrena el árbol leyendo los datos de un CSV
        try:
            print(f"Intentando cargar el dataset desde: {filepath}")
            data = pd.read_csv(filepath)
            print(f"Dataset cargado desde {filepath}.")
            X = data[["Velocidad", "Distancia"]].values
            y = data["Salto"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.arbol.fit(X_train, y_train)
            self.entrenado = True
            print("Árbol de decisiones entrenado correctamente.")
            self.guardar_modelo("modelo_arbol.pkl")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}.")
        except pd.errors.EmptyDataError:
            print(f"El archivo {filepath} está vacío o tiene un formato incorrecto.")
        except Exception as e:
            print(f"Ocurrió un error al entrenar el árbol de decisiones: {e}")

    def cargar_modelo(self, filepath="modelo_arbol.pkl"):
        # Carga un árbol de decisiones entrenado desde disco
        try:
            self.arbol = joblib.load(filepath)
            self.entrenado = True
            print(f"Modelo cargado desde {filepath}.")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}.")

    def entrenar_red_desde_csv(self, filepath="dataset.csv", modelo_guardado="modelo_red.h5"):
        # Entrena la red neuronal leyendo datos desde un CSV
        try:
            print(f"Intentando cargar el dataset desde: {filepath}")
            data = pd.read_csv(filepath)
            print(f"Dataset cargado desde {filepath}.")
            X = data[["Velocidad", "Distancia"]].values
            y = data["Salto"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.red_neuronal = Sequential([
                Dense(16, input_dim=2, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            self.red_neuronal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.red_neuronal.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
            self.red_neuronal.save(modelo_guardado)
            self.entrenado = True
            print(f"Modelo guardado en {modelo_guardado}.")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}.")
        except pd.errors.EmptyDataError:
            print(f"El archivo {filepath} está vacío o tiene un formato incorrecto.")
        except Exception as e:
            print(f"Ocurrió un error al entrenar el modelo: {e}")

    def entrenar_red(self, datos):
        # Entrena la red neuronal con datos en memoria
        if len(datos) > 0:
            X = datos[:, :2]
            y = datos[:, 2]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.red_neuronal = Sequential([
                Dense(16, input_dim=2, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            self.red_neuronal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.red_neuronal.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
            self.entrenado = True
        else:
            print("No hay datos para entrenar.")


model_manager = ModelManager()


# ------------------------ Funciones de menú y juego ------------------------

def mostrar_menu():
    # Muestra el menú principal y permite seleccionar el modo de juego
    global modo_auto

    # Cargar y procesar la imagen de fondo del menú
    menu_fondo = Image.open('./assets/game/fondo2.png')
    menu_fondo = menu_fondo.resize((WIDTH, HEIGHT))
    fondo_blur = menu_fondo.filter(ImageFilter.GaussianBlur(100))
    enhancer = ImageEnhance.Brightness(fondo_blur)
    fondo_oscuro = enhancer.enhance(0.5)
    fondo_surface = pygame.image.fromstring(fondo_oscuro.tobytes(), fondo_oscuro.size, fondo_oscuro.mode)
    pantalla.blit(fondo_surface, (0, 0))

    fuente_titulo = pygame.font.Font(None, 48)
    fuente_opciones = pygame.font.Font(None, 36)

    # Título del menú
    titulo = fuente_titulo.render("Menú Principal", True, BLANCO)
    pantalla.blit(titulo, (WIDTH // 2 - titulo.get_width() // 2, HEIGHT // 2 - 140))

    # Opciones del menú
    lineas_texto = [
        "Presiona 'M': Modo Manual",
        "Presiona 'A': Modo Automático (Red Neuronal)",
        "Presiona 'W': Modo Automático (Árbol de Decisiones)",
        "Presiona 'R': Entrenar Árbol",
        "Presiona 'E': Entrenar Red Neuronal",
        "Presiona 'D': Guardar Dataset en CSV",
        "Presiona 'Q': Salir"
    ]

    y = HEIGHT // 2 - 100
    for linea in lineas_texto:
        texto = fuente_opciones.render(linea, True, BLANCO)
        pantalla.blit(texto, (WIDTH // 2 - texto.get_width() // 2, y))
        y += texto.get_height() + 10

    pygame.display.flip()

    # Esperar a que el usuario seleccione una opción
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = "red"
                    model_manager.cargar_modelo_h5()
                    return
                if evento.key == pygame.K_w:
                    modo_auto = "arbol"
                    model_manager.cargar_modelo("modelo_arbol.pkl")
                    return
                if evento.key == pygame.K_m:
                    modo_auto = "manual"
                    return
                if evento.key == pygame.K_r:
                    model_manager.entrenar_arbol_desde_csv("dataset.csv")
                if evento.key == pygame.K_e:
                    model_manager.entrenar_red_desde_csv()
                if evento.key == pygame.K_d:
                    guardar_dataset()
                if evento.key == pygame.K_q:
                    pygame.quit()
                    exit()


def guardar_dataset(filepath="dataset.csv"):
    # Guarda los datos recopilados en un archivo CSV
    global datos_para_csv
    if len(datos_para_csv) == 0:
        print("No hay datos para guardar.")
        return

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Velocidad", "Distancia", "Salto"])
        writer.writerows(datos_para_csv)

    print(f"Dataset guardado en {filepath}.")
    datos_para_csv = []
    print("Lista de datos para CSV limpiada.")


def disparar_bala():
    # Inicia el movimiento de la bala con una velocidad aleatoria
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)
        bala_disparada = True


def reset_bala():
    # Devuelve la bala a su posición inicial
    global bala, bala_disparada
    bala.x = WIDTH - 50
    bala_disparada = False


def manejar_salto():
    # Controla el salto del jugador
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        if salto_altura > 0:
            # Fase ascendente
            jugador.y -= salto_altura
            salto_altura -= gravedad
        else:
            # Fase descendente
            jugador.y += abs(salto_altura)
            salto_altura -= gravedad

        # Si el jugador toca el suelo, se restablece el estado
        if jugador.y >= HEIGHT - 100:
            jugador.y = HEIGHT - 100
            salto = False
            en_suelo = True
            salto_altura = 15

    # Evitar que el jugador suba demasiado
    if jugador.y < 0:
        jugador.y = 0


def guardar_datos():
    # Guarda los datos de la jugada para el dataset
    global jugador, bala, velocidad_bala, salto, datos_para_csv
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0
    print(f"Guardando datos: Velocidad={velocidad_bala}, Distancia={distancia}, Salto={salto_hecho}")
    datos_para_csv.append((velocidad_bala, distancia, salto_hecho))


def jugar_automatico():
    # Controla el modo automático usando el modelo entrenado (red neuronal o árbol)
    global salto, en_suelo, velocidad_bala, jugador, bala, modo_auto

    distancia = abs(jugador.x - bala.x)
    if modo_auto == "red":
        if model_manager.entrenado and model_manager.red_neuronal:
            prediccion = model_manager.predecir_red(velocidad_bala, distancia)
        else:
            print("Red Neuronal no entrenada o no cargada.")
            return
    elif modo_auto == "arbol":
        if model_manager.entrenado:
            prediccion = model_manager.predecir_arbol(velocidad_bala, distancia)
        else:
            print("Árbol de Decisiones no entrenado o no cargado.")
            return
    else:
        print("Modo automático desconocido.")
        return

    # Si el modelo predice 1 y el jugador está en el suelo, se salta
    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False


def perder_y_regresar_menu():
    # Cuando se pierde, se guarda el dataset y se vuelve al menú
    global jugador, bala, nave, bala_disparada, salto, en_suelo
    print("Regresando al menú...")
    guardar_dataset()

    # Reiniciar estado
    jugador.x, jugador.y = 50, HEIGHT - 100
    bala.x = WIDTH - 50
    nave.x, nave.y = WIDTH - 100, HEIGHT - 100
    bala_disparada = False
    salto = False
    en_suelo = True

    mostrar_menu()


def reiniciar_juego():
    # Reinicia el juego después de una colisión
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, HEIGHT - 100
    bala.x = WIDTH - 50
    nave.x, nave.y = WIDTH - 100, HEIGHT - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    print("Datos recopilados para el modelo: ", datos_para_csv)
    mostrar_menu()


def update():
    # Actualiza la posición del fondo, jugador, bala y detecta colisiones
    global fondo_x1, fondo_x2, bala, velocidad_bala

    # Mover el fondo para simular desplazamiento
    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -WIDTH:
        fondo_x1 = WIDTH
    if fondo_x2 <= -WIDTH:
        fondo_x2 = WIDTH

    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Dibujar el jugador (por ahora un solo frame)
    pantalla.blit(jugador_frames[0], (jugador.x, jugador.y))

    # Si la bala no ha sido disparada, dispararla
    if not bala_disparada:
        disparar_bala()

    # Mover la bala
    bala.x += velocidad_bala
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Detectar colisión jugador-bala
    if jugador.colliderect(bala):
        print("¡Colisión detectada!")
        reiniciar_juego()

    # Dibujar la nave (UFO)
    pantalla.blit(nave_img, (nave.x, nave.y))


def pausa_juego():
    # Pausa o reanuda el juego
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
    else:
        print("Juego reanudado.")


def main():
    # Función principal del juego
    global salto, en_suelo, bala_disparada, modo_auto, datos_para_csv, pausa

    reloj = pygame.time.Clock()

    # Mostrar el menú inicial
    mostrar_menu()

    corriendo = True
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Pausar o reanudar el juego
                    pausa_juego()
                if evento.key == pygame.K_o:  # Guardar el dataset y regresar al menú
                    perder_y_regresar_menu()

        if not pausa:
            # Si es modo automático, usar el modelo para saltar
            if modo_auto == "red" or modo_auto == "arbol":
                jugar_automatico()
            # En modo manual, capturar datos para entrenar luego
            elif modo_auto == "manual":
                guardar_datos()

            manejar_salto()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
