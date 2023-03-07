import numpy as np
from math import sqrt

def l3_norm(elem1, elem2):
    tmp1 = elem1[0] - elem2[0]
    tmp2 = elem1[1] - elem2[1]
    tmp3 = elem1[2] - elem2[2]

    tmp1 *= tmp1
    tmp2 *= tmp2
    tmp3 *= tmp3

    return sqrt(tmp1 + tmp2 + tmp3)

def DTW(seqA, seqB, normalized=True):

    num_rows = len(seqA)
    num_cols = len(seqB)
    cost = np.zeros((num_rows, num_cols), dtype=np.float64)

    cost[0, 0] = l3_norm(seqA[0], seqB[0])

    for i in range(1, num_rows):
        cost[i, 0] = cost[i - 1, 0] + l3_norm(seqA[i], seqB[0])
    for j in range(1, num_cols):
        cost[0, j] = cost[0, j - 1] + l3_norm(seqA[0], seqB[j])

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l3_norm(seqA[i], seqB[j])

    if (normalized):
        return cost[-1, -1] / (num_rows + num_cols)
    else:
        return cost[-1, -1]
