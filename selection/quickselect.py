# selection/quickselect.py
import random

def partition(arr, left, right, pivot_index):
    pivot = arr[pivot_index]
    # mover pivot al final
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    store = left

    for i in range(left, right):
        if arr[i] > pivot:   # top-k busca los mayores
            arr[store], arr[i] = arr[i], arr[store]
            store += 1

    arr[right], arr[store] = arr[store], arr[right]
    return store

def quickselect(arr, k):
    """
    Retorna el top-k de arr usando Quickselect median-of-3.
    """
    arr = arr.copy()
    left, right = 0, len(arr) - 1
    k_index = k - 1

    while True:
        if left == right:
            return arr[:k]

        pivot_index = random.randint(left, right)
        pivot_index = partition(arr, left, right, pivot_index)

        if pivot_index == k_index:
            return arr[:k]
        elif pivot_index < k_index:
            left = pivot_index + 1
        else:
            right = pivot_index - 1
