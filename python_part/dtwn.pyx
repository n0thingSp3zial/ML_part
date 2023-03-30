import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from cpython cimport bool
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
cdef DTYPE_t l1_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])

    tmp1 *= tmp1

    return sqrt(tmp1)

@cython.boundscheck(False)
cdef DTYPE_t l2_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])
    cdef DTYPE_t tmp2 = (elem1[1] - elem2[1])

    tmp1 *= tmp1
    tmp2 *= tmp2

    return sqrt(tmp1 + tmp2)

@cython.boundscheck(False)
cdef DTYPE_t l3_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])
    cdef DTYPE_t tmp2 = (elem1[1] - elem2[1])
    cdef DTYPE_t tmp3 = (elem1[2] - elem2[2])

    tmp1 *= tmp1
    tmp2 *= tmp2
    tmp3 *= tmp3

    return sqrt(tmp1 + tmp2 + tmp3)

@cython.boundscheck(False)
cdef DTYPE_t l4_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])
    cdef DTYPE_t tmp2 = (elem1[1] - elem2[1])
    cdef DTYPE_t tmp3 = (elem1[2] - elem2[2])
    cdef DTYPE_t tmp4 = (elem1[3] - elem2[3])

    tmp1 *= tmp1
    tmp2 *= tmp2
    tmp3 *= tmp3
    tmp4 *= tmp4

    return sqrt(tmp1 + tmp2 + tmp3 + tmp4)

@cython.boundscheck(False)
cdef DTYPE_t l6_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])
    cdef DTYPE_t tmp2 = (elem1[1] - elem2[1])
    cdef DTYPE_t tmp3 = (elem1[2] - elem2[2])
    cdef DTYPE_t tmp4 = (elem1[3] - elem2[3])
    cdef DTYPE_t tmp5 = (elem1[4] - elem2[4])
    cdef DTYPE_t tmp6 = (elem1[5] - elem2[5])

    tmp1 *= tmp1
    tmp2 *= tmp2
    tmp3 *= tmp3
    tmp4 *= tmp4
    tmp5 *= tmp5
    tmp6 *= tmp6

    return sqrt(tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6)

@cython.boundscheck(False)
cdef DTYPE_t l8_norm(list elem1, list elem2):
    '''Function, which calculates L3 norm
    Calculates norm between two elements in L3
    :param elem1: A first element in L3
    :param elem2: A second element in L3
    :return: A norm ||elem1 - elem2||
    '''
    cdef DTYPE_t tmp1 = (elem1[0] - elem2[0])
    cdef DTYPE_t tmp2 = (elem1[1] - elem2[1])
    cdef DTYPE_t tmp3 = (elem1[2] - elem2[2])
    cdef DTYPE_t tmp4 = (elem1[3] - elem2[3])
    cdef DTYPE_t tmp5 = (elem1[4] - elem2[4])
    cdef DTYPE_t tmp6 = (elem1[5] - elem2[5])
    cdef DTYPE_t tmp7 = (elem1[6] - elem2[6])
    cdef DTYPE_t tmp8 = (elem1[7] - elem2[7])

    tmp1 *= tmp1
    tmp2 *= tmp2
    tmp3 *= tmp3
    tmp4 *= tmp4
    tmp5 *= tmp5
    tmp6 *= tmp6
    tmp7 *= tmp7
    tmp8 *= tmp8

    return sqrt(tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8)

@cython.boundscheck(False)
def DTW(list seqA, list seqB, bool normalized = True):
    '''Calculates DTW distance between two time sequences
    Calculates distance between two sequences by dynamic time warping
    :param seqA: A first time sequence (list), each element consists of list of three elements [x, y, z]
    :param seqB: A second time sequence (list), which is similar to the first
    :param normalized: Should be the distance normalized
    :return: A distance between two sequences
    '''
    cdef unsigned int numRows = len(seqA), numCols = len(seqB)
    cdef np.ndarray[DTYPE_t, ndim = 2, mode = "c"] cost = np.zeros((numRows, numCols), dtype = DTYPE)

    cdef unsigned int i, j

    if len(seqA[0]) == 3:

        cost[0, 0] = l3_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l3_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l3_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l3_norm(seqA[i], seqB[j])

    elif len(seqA[0]) == 6:

        cost[0, 0] = l6_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l6_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l6_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l6_norm(seqA[i], seqB[j])

    elif len(seqA[0]) == 8:

        cost[0, 0] = l8_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l8_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l8_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l8_norm(seqA[i], seqB[j])

    elif len(seqA[0]) == 2:

        cost[0, 0] = l2_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l2_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l2_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l2_norm(seqA[i], seqB[j])

    elif len(seqA[0]) == 4:

        cost[0, 0] = l4_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l4_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l4_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l4_norm(seqA[i], seqB[j])

    elif len(seqA[0]) == 1:

        cost[0, 0] = l1_norm(seqA[0], seqB[0])

        for i in range(1, numRows):
            cost[i, 0] = cost[i - 1, 0] + l1_norm(seqA[i], seqB[0])
        for j in range(1, numCols):
            cost[0, j] = cost[0, j - 1] + l1_norm(seqA[0], seqB[j])

        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]) + l1_norm(seqA[i], seqB[j])

    if (normalized):
        return cost[-1, -1] / (numRows + numCols)
    else:
        return cost[-1, -1]
