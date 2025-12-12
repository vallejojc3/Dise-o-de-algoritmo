# RA1 - Informe

## 1. Introducción
Breve descripción del problema y dataset (MNIST subset usado).

## 2. Arquitectura
- Entrada: D
- Capa oculta: h (ReLU)
- Salida: C (softmax)
- Parámetros: W1 (D×h), W2 (h×C)

## 3. Análisis asintótico
Detalles de forward/backprop: O(D·h + h·C) por muestra.
Coste por epoch: O(N·(D·h + h·C))

## 4. Experimentos
- Setup: learning rate, batch, epochs
- Tabla de resultados: tiempo vs B, val_acc
- Gráficas: tiempo/epoch vs B; val_acc vs epoch

## 5. Conclusiones
Discusión de trade-offs.

