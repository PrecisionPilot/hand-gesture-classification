import numpy as np
import math

def normalize_vector(nd_arr: np.ndarray) -> np.ndarray:
    vector_length = sum([i ** 2 for i in nd_arr]) ** 0.5
    normalized_arr = [i / vector_length for i in nd_arr]
    return normalized_arr

arr = np.array([[3, 4], [5, 3]])
noramlized_arr = [normalize_vector(row) for row in arr]
print(noramlized_arr)