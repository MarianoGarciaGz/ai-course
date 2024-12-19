import pygame
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageFilter, ImageEnhance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import joblib

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

fuente = pygame.font.Font(None, 36)

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

jugador_frames = [
    pygame.image.load('./assets/sprites/mono_frame_1.png'),
    pygame.image.load('./assets/sprites/mono_frame_2.png')
]
bala_img = pygame.image.load('./assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('./assets/game/fondo2.png')
nave_img = pygame.image.load('./assets/game/ufo.png')

fondo_img = pygame.transform.scale(fondo_img, (w, h))

jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)

salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

velocidad_bala = -10
bala_disparada = False

fondo_x1 = 0
fondo_x2 = w

pausa = False
menu_activo = True
modo_auto = False

datos_para_csv = []  # Datos en memoria por sesión

class ModelManager:
    def __init__(self):
        self.arbol = DecisionTreeClassifier()
        self.red_neuronal = None
        self.entrenado = False

    def cargar_modelo_h5(self, filepath="modelo_red.h5"):
        try:
            self.red_neuronal = keras.models.load_model(filepath)
            self.red_neuronal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.entrenado = True
            print(f"Modelo de red neuronal cargado desde {filepath}.")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}. Por favor, entrena y guarda un modelo primero.")
            self.entrenado = False

    def predecir_red(self, velocidad, distancia, umbral=0.5):
        if not self.entrenado or self.red_neuronal is None:
            raise ValueError("El modelo de red neuronal no está entrenado.")

        entrada = np.array([[velocidad, distancia]])
        prediccion = self.red_neuronal.predict(entrada)[0][0]

        return 1 if prediccion >= umbral else 0

    def entrenar_arbol(self, datos):
        self.arbol = DecisionTreeClassifier()
        self.entrenado = False
        if len(datos) > 0:
            X, y = datos[:, :2], datos[:, 2]
            self.arbol.fit(X, y)
            self.entrenado = True
            print("Árbol entrenado correctamente.")
        else:
            print("No hay datos para entrenar el árbol.")

    def predecir_arbol(self, velocidad, distancia):
        if not self.entrenado:
            raise ValueError("El modelo no está entrenado.")
        return int(self.arbol.predict([[velocidad, distancia]])[0])

    def guardar_modelo_arbol(self, filepath="modelo_arbol.pkl"):
        if self.entrenado:
            joblib.dump(self.arbol, filepath)
            print(f"Modelo de árbol guardado en {filepath}.")
        else:
            print("El modelo de árbol no está entrenado, no se guardará.")

    def entrenar_red(self, datos):
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
            self.red_neuronal.save("modelo_red.h5")
            print("Red neuronal entrenada y guardada en modelo_red.h5.")
        else:
            print("No hay datos para entrenar la red neuronal.")

    def cargar_modelo(self, filepath="modelo_arbol.pkl"):
        try:
            self.arbol = joblib.load(filepath)
            self.entrenado = True
            print(f"Modelo de árbol cargado desde {filepath}.")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filepath}.")

    def esta_entrenado(self):
        return self.entrenado

model_manager = ModelManager()

def mostrar_menu():
    global modo_auto

    GRIS = (37, 46, 128)
    pantalla.fill(GRIS)

    fuente_titulo = pygame.font.SysFont('courier', 40)
    fuente_opciones = pygame.font.SysFont('courier', 28)

    titulo = fuente_titulo.render("Menú Principal", True, BLANCO)
    pantalla.blit(titulo, (10, 10))

    # Ajustamos las opciones del menú:
    # 1 = Modo Manual
    # 2 = Modo Automático (Red Neuronal)
    # 3 = Modo Automático (Árbol de Decisiones)
    # 4 = Entrenar Modelos (Árbol + Red)
    # 0 = Salir
    lineas_texto = [
        "1 = Modo Manual",
        "2 = Modo Automático (Red Neuronal)",
        "3 = Modo Automático (Árbol de Decisiones)",
        "4 = Entrenar Modelos (Árbol + Red)",
        "0 = Salir"
    ]

    y = 60
    for linea in lineas_texto:
        texto = fuente_opciones.render(linea, True, BLANCO)
        pantalla.blit(texto, (10, y))
        y += texto.get_height() + 10

    pygame.display.flip()

    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:
                    modo_auto = "manual"
                    return
                if evento.key == pygame.K_2:
                    modo_auto = "red"
                    model_manager.cargar_modelo_h5()
                    return
                if evento.key == pygame.K_3:
                    modo_auto = "arbol"
                    model_manager.cargar_modelo("modelo_arbol.pkl")
                    return
                if evento.key == pygame.K_4:
                    entrenar_ambos_modelos_desde_memoria()
                if evento.key == pygame.K_0:
                    pygame.quit()
                    exit()

def guardar_datos_en_csv(filepath="dataset.csv"):
    global datos_para_csv
    import csv

    if not datos_para_csv:
        print("No hay datos para guardar.")
        return

    # Escribir datos en un archivo CSV
    columnas = ['velocidad', 'distancia', 'salto']
    try:
        # Si el archivo no existe, crea uno nuevo con encabezados
        archivo_existe = os.path.exists(filepath)
        with open(filepath, mode='a', newline='') as archivo_csv:
            escritor = csv.writer(archivo_csv)
            if not archivo_existe:
                escritor.writerow(columnas)  # Escribe encabezados
            escritor.writerows(datos_para_csv)
        print(f"Datos guardados en {filepath}.")
    except Exception as e:
        print(f"Error al guardar datos en CSV: {e}")

def entrenar_ambos_modelos_desde_memoria():
    global datos_para_csv, model_manager
    datos = np.array(datos_para_csv)
    if len(datos) == 0:
        print("No hay datos suficientes para entrenar los modelos.")
        return
    # Entrenamos el árbol
    model_manager.entrenar_arbol(datos)
    model_manager.guardar_modelo_arbol()

    # Entrenamos la red
    model_manager.entrenar_red(datos)

def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)
        bala_disparada = True

def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50
    bala_disparada = False

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        if salto_altura > 0:
            jugador.y -= salto_altura
            salto_altura -= gravedad
        else:
            jugador.y += abs(salto_altura)
            salto_altura -= gravedad

        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            en_suelo = True
            salto_altura = 15

    if jugador.y < 0:
        jugador.y = 0

def guardar_datos():
    global jugador, bala, velocidad_bala, salto, datos_para_csv
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0

    datos_para_csv.append((velocidad_bala, distancia, salto_hecho))

def jugar_automatico():
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

    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False

def perder_y_regresar_menu():
    global datos_para_csv, jugador, bala, nave, bala_disparada, salto, en_suelo, modo_auto

    if modo_auto == "manual" and datos_para_csv:
        # Guardar datos en CSV antes de reiniciar
        guardar_datos_en_csv()

    # Reiniciar el juego
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    mostrar_menu()


def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    mostrar_menu()

def update():
    global fondo_x1, fondo_x2, bala, velocidad_bala

    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    pantalla.blit(jugador_frames[0], (jugador.x, jugador.y))

    if not bala_disparada:
        disparar_bala()

    bala.x += velocidad_bala

    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    if jugador.colliderect(bala):
        print("¡Colisión detectada!")
        reiniciar_juego()

    pantalla.blit(nave_img, (nave.x, nave.y))

def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
    else:
        print("Juego reanudado.")

def main():
    global salto, en_suelo, bala_disparada, modo_auto, datos_para_csv, pausa, model_manager

    # Re-inicializar modelos al inicio de la sesión
    model_manager = ModelManager()
    datos_para_csv = []  # Limpiar datos para la nueva sesión

    reloj = pygame.time.Clock()

    mostrar_menu()

    correr = True
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_o:
                    # Entrenar modelos y regresar al menú
                    perder_y_regresar_menu()

        if not pausa:
            if modo_auto == "red" or modo_auto == "arbol":
                jugar_automatico()
            elif modo_auto == "manual":
                guardar_datos()
            manejar_salto()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

def perder_y_regresar_menu():
    global datos_para_csv, jugador, bala, nave, bala_disparada, salto, en_suelo, modo_auto

    if modo_auto == "manual":
        # Entrenar ambos modelos con los datos en memoria solo si el modo es manual
        entrenar_ambos_modelos_desde_memoria()

    # Reiniciamos el juego
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    mostrar_menu()

def guardar_datos():
    global jugador, bala, velocidad_bala, salto, datos_para_csv, modo_auto

    if modo_auto != "manual":
        # No guardar datos si no estamos en modo manual
        return

    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0

    datos_para_csv.append((velocidad_bala, distancia, salto_hecho))

if __name__ == "__main__":
    main()
