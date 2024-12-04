# Respuestas

## **¿Son los datos linealmente separables?**

No, los datos **no son linealmente separables**, ya que no es posible dividir las tres clases de riesgo con una línea recta o un plano en el espacio de características.

---

## **¿Qué ajustes podrían hacer al modelo para mejorar la clasificación?**

1. Incrementar el tamaño del conjunto de datos para una mejor generalización.
2. Añadir más neuronas o capas ocultas para capturar relaciones más complejas.
3. Aplicar regularización \(L2\) o dropout para prevenir el sobreajuste.
4. Ajustar la tasa de aprendizaje o implementar "early stopping".
5. Probar funciones de activación como ReLU en la capa oculta.

---

## **Describir cada una de las partes del modelo implementado:**

1. **Entrada:**
   - Características: Historial de pagos, ingresos mensuales, relación deuda-ingreso.

2. **Capa Oculta:**
   - **Neuronas:** 4
   - **Función de Activación:** Sigmoid (\( \sigma(z) = \frac{1}{1 + e^{-z}} \)) para introducir no linealidad.

3. **Capa de Salida:**
   - **Neuronas:** 3 (una por clase).
   - **Función de Activación:** Softmax para convertir las salidas en probabilidades.

4. **Función de Pérdida:**
   - **Entropía Cruzada Categórica:** Mide la discrepancia entre las predicciones y las etiquetas reales.

5. **Optimización:**
   - Algoritmo: Descenso de gradiente estocástico.
   - Parámetros ajustados: Pesos y sesgos (\( W1, b1, W2, b2 \)).

6. **Entrenamiento:**
   - Incluye propagación hacia adelante, cálculo de pérdida, retropropagación y actualización de parámetros.
