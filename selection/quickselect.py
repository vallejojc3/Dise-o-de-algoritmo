# selection/quickselect.py
import random

def quickselect(arr, k):
    """
    Devuelve el k-ésimo más pequeño (0-index).
    """
    if len(arr) == 1:
        return arr[0]
    pivot = random.choice(arr)
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

def median(arr):
    n = len(arr)
    if n % 2 == 1:
        return quickselect(arr, n//2)
    else:
        a = quickselect(arr, n//2 - 1)
        b = quickselect(arr, n//2)
        return 0.5 * (a + b)
