# experiments/graph_epochs.py

import matplotlib.pyplot as plt
from experiments.exp_epochs import experiment

def graph_epochs():
    results = experiment()

    epochs = list(results.keys())
    times = [results[e][0] for e in epochs]
    accs = [results[e][1] for e in epochs]

    # ---- Tiempo ----
    plt.figure(figsize=(8,5))
    plt.plot(epochs, times, marker="o")
    plt.xlabel("Épocas")
    plt.ylabel("Tiempo (s)")
    plt.title("Tiempo según número de épocas")
    plt.grid(True)
    plt.savefig("epochs_time.png")
    print("Imagen guardada: epochs_time.png")

    # ---- Accuracy ----
    plt.figure(figsize=(8,5))
    plt.plot(epochs, accs, marker="o", color="orange")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.title("Accuracy según número de épocas")
    plt.grid(True)
    plt.savefig("epochs_accuracy.png")
    print("Imagen guardada: epochs_accuracy.png")

if __name__ == "__main__":
    graph_epochs()
