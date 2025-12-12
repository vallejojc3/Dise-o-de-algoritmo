# experiments/graph_ra3.py

import matplotlib.pyplot as plt
from experiments.exp_ra3 import experiment

def graph_ra3():
    t_norm, t_struct = experiment()

    plt.bar(["Normal", "Con estructuras"], [t_norm, t_struct])
    plt.ylabel("Tiempo (s)")
    plt.title("Impacto de estructuras de datos en el entrenamiento")
    plt.savefig("ra3_comparison.png")
    print("Imagen guardada: ra3_comparison.png")

if __name__ == "__main__":
    graph_ra3()
