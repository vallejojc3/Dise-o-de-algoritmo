# experiments/graph_hidden_size.py

import matplotlib.pyplot as plt
from experiments.exp_hidden_size import experiment

def graph_hidden_size():
    results = experiment()
    h_values = list(results.keys())
    times = [results[h][0] for h in h_values]
    accuracies = [results[h][1] for h in h_values]

    # ---- Tiempo ----
    plt.figure(figsize=(8,5))
    plt.plot(h_values, times, marker='o')
    plt.title("Tiempo según Número de Neuronas Ocultas (h)")
    plt.xlabel("Número de neuronas (h)")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)
    plt.savefig("hidden_size_time.png")
    print("Imagen guardada: hidden_size_time.png")

    # ---- Accuracy ----
    plt.figure(figsize=(8,5))
    plt.plot(h_values, accuracies, marker='o', color='orange')
    plt.title("Accuracy según Número de Neuronas Ocultas (h)")
    plt.xlabel("Número de neuronas (h)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("hidden_size_accuracy.png")
    print("Imagen guardada: hidden_size_accuracy.png")

if __name__ == "__main__":
    graph_hidden_size()
